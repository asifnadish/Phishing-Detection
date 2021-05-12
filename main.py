# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:40:30 2019

@author: Asif Iqbal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import urllib.request
from urllib.parse import urlparse
from urllib.error import HTTPError
from bs4 import BeautifulSoup
import requests 
import re
dataset = pd.read_csv('trainingData.csv')
X = dataset.iloc[:, [0,1,2,3,4,5,6,10,11,12,13,15,16,17,25]].values  #  15 features
#X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 30].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

'''
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)    '''

'''
#Fitting K-NN to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5 , metric = 'minkowski',p=2)
classifier.fit(X_train,y_train)    '''  

'''
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)     '''

'''
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)     '''

'''
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)       '''

'''
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)       '''

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)  



# Making the Confusion Matrix  to findout the number of correct and incorrect prediction
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

classifier.score(X_train,y_train)

classifier.score(X_test,y_test)


# K-fold cross validation
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
accuracy.mean()     





class Feature_Extration:
    def _init_(self):
        pass
        
    def getProtocol(self,url):
        return urlparse(url).scheme
    
    
    def getHostName(self,url):
        return urlparse(url).hostname
    
    
    def getPath(self,url):
        return urlparse(url).path
    
    
    def long_URL(self,url):
        if len(url)<54:
            return -1
        elif len(url) >=54 and len(url) <=75:
            return 0
        return 1
    
    def have_at_symbol(self,url):
        if "@" in url:
            return 1
        return -1
        
    def redirection(self,url):
        """If the url has symbol(//) after protocol then such URL is to be classified as phishing """
        if "//" in urlparse(url).path:
            return 1
        return -1

    def prefix_suffix_seperation(self,url):
        if '-' in urlparse(url).netloc:
            return 1
        return -1
    
    def having_ip_address(self,url):
        match=re.search('(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  #IPv4
                        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'  #IPv4 in hexadecimal
                        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}',url)     #Ipv6
        if match:
            return 1
        else:
            return -1


    def shortening_service(self,url):
        match=re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                        'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                        'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                        'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                        'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                        'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                        'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net',url)
        if match:
            return 1
        else:
            return -1
        
    def get_domain_name(self,url):
         try:
            results = urlparse(url).hostname.split('.')
            return results[-2] + '.' + results[-1]
         except:
             return ''
         
   
    
    def get_links(self,url):
       
        page = requests.get(url)
        soup = BeautifulSoup(page.content,'html.parser')
        href_links = soup.find_all('a',href=True)
        src_links = soup.find_all('img',src=True)
        action_links = soup.find_all('form')
        for link in href_links:
            attr = link.get('href')
            if attr[0] == '/' :
                attr = url + attr
            elif attr[0] == '#':
                attr = url
            elif attr == 'javascript:void(0)':
                attr = url    
            elif attr[0] == '.':
                attr = url
            hlinks.append(attr)
            '''
            elif attr.find(url) == -1:
                attr = url + attr   '''
           
            
        for link in src_links:
            attr = link.get('src')
            if attr[0] == '/':
                attr = url + attr    
            slinks.append(attr)
       
        for link in action_links:
            attr = link.get('action')
            if attr[0] == '/':
                attr = url + attr    
            alinks.append(attr)
           
        
    def redirect(self,url):
        rcount = 0
        
        for link in hlinks:
             page = requests.get(link,timeout=300)  #disabling validation
             if 300 <= page.status_code <400 :
                 rcount = rcount + 1
        if rcount <= 1:
            return -1
        elif 2 <= rcount < 4:
            return 0
        return 1
    
    
    def get_null_links(self,url):
        ncount = 0
        count = 0
       
        for link in hlinks:
            count = count + 1
            if link.find('#') != -1:
                ncount = ncount + 1
            if link.find('JavaScript:void(0)') != -1:
                ncount = ncount + 1
            if len(link) == 0:
                ncount = ncount + 1
        if count != 0:            
            nratio = ncount / count 
        else:
            nratio = 0
        if nratio < 0.31:
            return -1
        elif 0.31 <= nratio < 0.67:
            return 0
        return 1
    def get_action_value(self,url):
       
        for attr in alinks:
            if attr.find('#') != -1:
                return 1
            if attr.find('.php') != -1:
                return 1
            if attr.find('JavaScript:void(0)') != -1:
                return 1
            if len(attr) == 0:
                return 1
            return -1
    def having_sub_domain(self,url):
        host_name = urlparse(url).hostname
        dcount = host_name.count('.')
        if dcount <= 2:
            return -1
        elif dcount == 3:
            return 0
        else:
            return 1
    def having_favicon(self,url):
        domain_name = self.get_domain_name(url)
        page = requests.get(url)
        soup = BeautifulSoup(page.content,'html.parser')
        link_attr = soup.find_all('link')
        
    def having_port_open(self,url):
        parsed_url = urlparse(url).netloc
        
        if ":80" in url or ":443" in parsed_url:
            return -1
        else:
            port = re.search(":[0-9]", parsed_url)
            if port:
                return  1
            else:
                return  -1
    def https_in_host_part(self,url):
        parsed_url = urlparse(url).hostname
        if 'https' in parsed_url:
            return 1
        else:
            return -1
    def request_url(self,url):
        nrcount=0
        count=0
        domain_name = self.get_domain_name(url)
        page = requests.get(url)
       
        for link in slinks:
            count = count +1
            if domain_name in link:
                nrcount = nrcount + 1
        rcount = count - nrcount
        if count == 0:
            ratio = 0
        else:
            ratio = rcount / count
        if ratio < 0.22:
            return -1
        elif 0.22 <= ratio < 0.61:
            return 0
        else:
            return 1
    def submitting_to_email(self,url):
        for attr in alinks:
            if 'mail()' in attr:
                return 1 
            if 'mailto:' in attr:
                return 1
            else:
                return -1
    def abnormal_url(self,url):
        host_name = urlparse(url).hostname
        if host_name == '':
            return 1
        else:
            return -1
    def web_traffic(self,url):
        try:
            rank = BeautifulSoup(urllib.request.urlopen("http://data.alexa.com/data?cli=10&dat=s&url=" + url).read(), "xml").find("REACH")['RANK']
        except TypeError:
            return 1
        except HTTPError:
            return 0
        rank= int(rank)
        if (rank<100000):
            return -1
        else:
            return 0
                
        


            
        

        
hlinks = []         #href links
slinks = []         #src links
alinks = []         #action links            
           
            
         
protocol = []               #1       
hostname = []               #2
path = []                   #3
url_size = []               #4
have_at_symbol = []         #5
redirected = []             #6
having_dash_in_host = []    #7
having_ip = []              #8
tiny_url = []               #9
#no_hyperlinks = []          #10
#internal_link_ratio = []    #11
redirection = []            #11
null_hyperlinks = []        #12
SFH = []                    #13
having_sub_domain = []      #14
port = []                   #15
https_token = []             #16
requests_url = []            #17
submitting_data_to_email = [] #18
abnormal_url = []           #19
traffic = []                #20

#K fold validation


obj = Feature_Extration()
url = input('Enter the URL :')
obj.get_links(url)   
url_size.append(obj.long_URL(url))
protocol.append(obj.getProtocol(url))
hostname.append(obj.getHostName(url))
path.append(obj.getPath(url))
have_at_symbol.append(obj.have_at_symbol(url))
having_dash_in_host.append(obj.prefix_suffix_seperation(url))
redirected.append(obj.redirection(url))
having_ip.append(obj.having_ip_address(url))
tiny_url.append(obj.shortening_service(url))
#no_hyperlinks.append(obj.no_hyperlinks(url))
#internal_link_ratio.append(obj.internal_link(url))
null_hyperlinks.append(obj.get_null_links(url))
SFH.append(obj.get_action_value(url))
having_sub_domain.append(obj.having_sub_domain(url))
port.append(obj.having_port_open(url))
https_token.append(obj.https_in_host_part(url))
requests_url.append(obj.request_url(url))
submitting_data_to_email.append(obj.submitting_to_email(url))
abnormal_url.append(obj.abnormal_url(url))
traffic.append(obj.web_traffic(url))
#redirection.append(obj.redirect(url))     14,

if SFH[0] == None:
    SFH[0] = 1
if submitting_data_to_email[0] == None:
    submitting_data_to_email[0] = 1   

d = {'0':having_ip,'1':url_size,'2':tiny_url,'3':have_at_symbol,'4':redirected,'5':having_dash_in_host,'6':having_sub_domain,'7':port,'8':https_token,'9':requests_url,'10':null_hyperlinks,'11':SFH,'12':submitting_data_to_email,'13':abnormal_url,'14':traffic}
df = pd.DataFrame(d)             

result = classifier.predict(df) 
    

    
    
