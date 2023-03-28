import numpy as np
import pandas as pd
from pprint import pprint
import itertools
import operator
from sksurv.metrics import concordance_index_censored

class Patient:
    """Patient class

    Patientごとの生存時間解析を行うクラス
    
    Attributes:
        # 患者の情報
        age (int): 患者の年齢
        detection (int): 発見契機 symptoms detected = 0, screen detected = 1, unknown = 2
        size (float): Tumour size (mm)
        grade (int) : Tumour grade 1, 2 or 3 only
        nodes (int) : がんが転移したリンパ節の数
        er (int) : ER+ = 1, ER- = 0
        her2 (int) : her2+ = 1, her2- = 0 missing = 2
        ki67 (int) : ki67+ = 1, ki67- = 0 missing = 2
        event (int) : event発生の有無 0, 1
        duration (int) : event発生時間
        
        # Tretment Options
        chemo_gen (int) : Chemo generation 0, 2 or 3 only 
        horm (int) :  Hormone therapy Yes = 1, no = 0
        traz (int) : Trastuzumab therapy Yes = 1, no = 0
        bis (int) : Bisphosphonate therapy Yes = 1, no = 0
        treatment_options (str) : 患者が行っている治療法の結合 -> "c0:h1:t0:b1"
        
        # other
        max_time (int) : 打ち切りまでの時間（年）
        time 
        coef_ : 回帰係数
        _VERSION : predictのバージョン
    
    """
    def __init__(self,patient_data):
        # 患者の情報
        
        self.patient_data = patient_data
        self.pID =  patient_data["pID"].values[0]
        self.age =  patient_data["age"].values[0]
        
        self.detection = patient_data['detection'].values[0]
        self.detection =  0.204 if self.detection == 2 else self.detection
        
        self.size = patient_data['tumour_size'].values[0]
        
        self.grade = patient_data['tumour_grade'].values[0]
        self.grade = 2.13 if self.grade == 9 else self.grade
        
        self.nodes = patient_data['nodes'].values[0]
        self.er = patient_data['er'].values[0]
        self.her2 = patient_data['her2'].values[0]
        self.ki67 = patient_data['ki67'].values[0]
        
        # Tretment Options
        self.chemo_gen = patient_data['chemo_generation'].values[0]
        self.horm = patient_data['hormonetherapy'].values[0]
        self.traz = patient_data['trastuzumab'].values[0]
        self.bis = patient_data['bisphosphonates'].values[0]
        self.event = patient_data['event'].values[0]
        self.duration = patient_data['duration'].values[0]
        
        self.treatment_options = f"h{int(self.horm)}:c{int(self.chemo_gen)}:t{int(self.traz)}:b{int(self.bis)}"
        #self.treatment_options_forR = self.__treatment_options_forR()
        
        # other
        self.max_time = 15
        self.time = np.array(range(1,self.max_time+1))
        self.__set_model()
        self.__other_causes_model()
        self.__cancer_causes_model()
        
        self._VERSION = 'predict version 2.2 https://breast.predict.nhs.uk/ '

    def treatment_options_forR(self):
        options = f"h{int(self.horm)}:c{int(self.chemo_gen)}:t{int(self.traz)}:b{int(self.bis)}"
        treatments = [treat[0] for treat in options.split(":") if int(treat[1]) > 0]
        if len(treatments) == 0:
            return 's'
        else:
            treatment_options = ''
            for treatment in treatments:
                treatment_options += treatment
            return treatment_options
        
    def get_patient_data(self):
        """患者の入力情報を辞書式で返すメソッド

        Returns:
            dict: 患者の情報
        """
        columns = list(self.patient_data.columns)
        data = self.patient_data.values[0]
        param_dict = {col:datum for col, datum in zip(columns, data)}
        self.params = param_dict
        return param_dict
    
    def print_patient_data(self):
        pprint(self.get_patient_data())
    
    def __set_model(self):
        """患者の情報からモデルの重みを決定するメソッド

        Returns:
            np.array:重みの情報
        """
        if self.er == 1:
            # data_point
            age_mfp_1 = (self.age/10)**-2-0.0287449295 
            age_mfp_2 = (self.age/10)**-2*np.log(self.age/10)-0.0510121013 
            size_mfp = np.log(self.size/100)+1.545233938 
            nodes_mfp = np.log((self.nodes+1)/10)+1.387566896 
            grade_mfp =  self.grade
            detection_mfp = 0.204 if self.detection == 9 else self.detection
            her2_mfp = 1
            ki67_mfp = 1
            
            # weight
            age_beta_1 = 34.53642 
            age_beta_2 = -34.20342  
            size_beta = 0.7530729 
            nodes_beta = 0.7060723 
            grade_beta = 0.746655 
            detection_beta = -0.22763366
            her2_beta = 0.2413 if self.her2 == 1 else -0.0762 if self.her2 == 0 else 0
            ki67_beta = 0.14904 if self.ki67 == 1 else -0.11333 if self.ki67 == 0 else 0
            
            
        else:
            # data_point
            age_mfp_1 = self.age-56.3254902 
            age_mfp_2 = 0  
            size_mfp = (self.size/100)**0.5-0.5090456276  
            nodes_mfp = np.log((self.nodes+1)/10)+1.086916249   
            grade_mfp = 1 if self.grade == 2 or self.grade == 3 else 0
            detection_mfp = 0
            her2_mfp = 1
            ki67_mfp = 1
            
            # weight
            age_beta_1 = 0.0089827
            age_beta_2 = 0
            size_beta = 2.093446
            nodes_beta = 0.6260541
            grade_beta = 1.129091 
            detection_beta = 0
            her2_beta = 0.2413 if self.her2 == 1 else -0.0762 if self.her2 == 0 else 0
            ki67_beta = 0
            
        x_1 = np.array([age_mfp_1,age_mfp_2,detection_mfp,size_mfp,grade_mfp,nodes_mfp,her2_mfp,ki67_mfp])
        w_1 = np.array([age_beta_1,age_beta_2,detection_beta,size_beta,grade_beta,nodes_beta,her2_beta,ki67_beta])
        pi = np.dot(w_1,x_1)
 
        c_mfp = 1 
        h_mfp = 1 
        # h10_mfp = 1
        t_mfp = 1
        b_mfp = 1

        c_beta = 0 if self.chemo_gen == 0 else -0.248 if self.chemo_gen == 2 else -0.446
        h_beta = -0.3857 if self.horm == 1 and self.er == 1 else 0
        # h10_beta = [horm for i in range(10)] + [horm-0.26 for i in range(len(time)-10)]
        t_beta = -0.3567 if self.her2 == 1 and self.traz == 1 else 0
        b_beta = -0.198 if self.bis == 1 else 0

        x_2 = np.array([c_mfp,h_mfp,t_mfp,b_mfp])
        w_2 = np.array([c_beta,h_beta,t_beta,b_beta])
        rx = np.dot(w_2,x_2)
        
        self.coef_ = {"w_1": w_1, "x_1": x_1, "w_2": w_2, "x_2": x_2}
        self.base = "ER+" if self.er == 1 else "ER-"
        self.pi = pi
        self.rx = rx
        self.polynomial = pi + rx
    
    def __other_causes_model(self):
        """癌が原因でないモデル
        
        """
        # Other mortality prognostic index (mi)
        mi = 0.0698252*((self.age/10)**2-34.23391957)
        
        # Generate cumulative baseline other mortality
        base_m_cum_oth = np.exp(-6.052919 + (1.079863*np.log(self.time)) + (0.3255321*self.time**0.5))
        
        # Generate cumulative survival non-breast mortality
        s_cum_oth = np.exp(-np.exp(mi)*base_m_cum_oth)
        # Convert cumulative mortality rate into cumulative risk
        m_cum_oth = 1- s_cum_oth
        
        # Annual other mortality rate
        m_oth = np.array([i for i in m_cum_oth])

        for i in range(1,self.max_time):
            m_oth[i] = m_cum_oth[i] - m_cum_oth[i-1]
            
        self._s_cum_oth = s_cum_oth
        self._m_cum_oth = m_cum_oth
        
    def __cancer_causes_model(self):
        """癌が原因のモデル
        """
        # Generate cumulative baseline breast mortality
        if self.base == "ER+":
            base_m_cum_br = np.exp(0.7424402 - 7.527762/self.time**0.5 - 1.812513*np.log(self.time)/self.time**0.5)
        else:
            base_m_cum_br = np.exp(-1.156036 + 0.4707332/self.time**2 - 3.51355/self.time)

        self.cum_baseline_hazard_ = base_m_cum_br
        # Annual cancer mortality rate
        base_m_br = np.array([i for i in base_m_cum_br])

        for i in range(1,self.max_time):
            base_m_br[i] = base_m_cum_br[i] - base_m_cum_br[i-1]

        # Generate the annual breast cancer specific mortality rate
        m_br = base_m_br * np.exp(self.polynomial)
        
        #累積和
        m_cum_br = np.array(list(itertools.accumulate(m_br)))
        s_cum_br = np.exp(- m_cum_br)
        m_cum_br =  1- s_cum_br
        
        self._s_cum_br = s_cum_br
        self._m_cum_br = m_cum_br
    
    def predict(self):
        """risk_scoreを返すメソッド
        Returns
        -------
        risk_score : float
            Predicted risk scores.
        
        """
        return self.polynomial
        
    def predict_cumulative_hazard_function(self):
        """Predict cumulative hazard function.
        Returns
        -------
        cum_hazard : ndarray ()

        """
        
        m_cum_all = 1 - self._s_cum_oth*self._s_cum_br
        self._m_cum_all = m_cum_all*100
        return self._m_cum_all
        
    def predict_survival_function(self):
        """Predict cumulative survival function.
        Returns
        -------
        survival : ndarray ()

        """
        m_cum_all = 1 - self._s_cum_oth*self._s_cum_br
        s_cum_all = 100-100*m_cum_all
        
        self._s_cum_all = s_cum_all
        return self._s_cum_all
    

def concordance_score(input_data):
    """calculating concordance index

    Args:
        input_data (pd.DataFrame): input_data.csvのDataFrame

    Returns:
        float: concordance index
    """

    input_data = input_data
    patients = [Patient(patient_data=input_data.query(f" pID == {pID}")) for pID in input_data['pID'].unique()]
    data = [(patient.pID, patient.predict(),patient.event, patient.duration) for patient in patients]
    score_df = pd.DataFrame(data,columns=['pID', 'risk_score', 'event','duration']).reset_index(drop=True)
    score_df['event'] = score_df['event'].apply(lambda x: True if x == 1 else False)

    score = concordance_index_censored(
        event_indicator = score_df['event'].values,
        event_time = score_df['duration'].values,
        estimate = score_df['risk_score'].values,
    )[0]
        
    return score
    
