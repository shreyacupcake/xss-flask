import pandas as pd
import numpy as np
data=pd.read_csv("XSS_dataset_1.csv")

import numpy as np
from urllib.parse import unquote
from sklearn.feature_extraction.text import CountVectorizer
import nltk

nltk.download('punkt')

def extract_features_from_scripts(script_strings):
    features = []

    for script in script_strings:
        # Decode the script string and preprocess it
        decoded_script = unquote(script)
        decoded_script = decoded_script.replace(" ", "")
        lower_script = decoded_script.lower()

        # Feature 1: Count of specific HTML tags
        malicious_tags = ['<link', '<object', '<form', '<embed', '<layer', '<style', '<applet', '<meta',
                          '<img', '<iframe', '<input', '<body', '<video', '<button', '<math', '<svg',
                          '<div', '<a', '<frameset', '<table', '<comment', '<base', '<image']
        feature1 = sum(lower_script.count(tag) for tag in malicious_tags)

        # Feature 2: Count of specific malicious methods/events
        malicious_methods = ['exec', 'fromcharcode', 'eval', 'alert', 'getelementsbytagname', 'write', 'unescape',
                             'escape', 'prompt', 'onload', 'onclick', 'onerror', 'onpage', 'confirm', 'marquee']
        feature2 = sum(lower_script.count(method) for method in malicious_methods)

        # Feature 3: Count of ".js"
        feature3 = lower_script.count('.js')

        # Feature 4: Count of "javascript"
        feature4 = lower_script.count('javascript')

        # Feature 5: Length of the string
        feature5 = len(lower_script)

        # Feature 6: Count of "<script" with various encodings
        feature6 = lower_script.count('<script') + lower_script.count('&lt;script') + lower_script.count('%3cscript') + lower_script.count('%3c%73%63%72%69%70%74')

        # Feature 7: Count of special characters
        special_characters = ['&', '<', '>', '"', '\'', '/', '%', '*', ';', '+', '=', '%3C']
        feature7 = sum(lower_script.count(char) for char in special_characters)

        # Feature 8: Count of "http"
        feature8 = lower_script.count('http')

        # Feature 9: Count of "document.cookie"
        feature9 = lower_script.count('document.cookie')

        # Feature 10: Count of "window.location"
        feature10 = lower_script.count('window.location')

        # Feature 11: Count of "eval(" (JavaScript eval function)
        feature11 = lower_script.count('eval(')

        # Feature 12: Count of "innerHTML" (may be used to manipulate DOM)
        feature12 = lower_script.count('innerHTML')

        # Feature 13: Count of "src" attribute in HTML tags
        feature13 = lower_script.count('src=')

        # Feature 14: Count of "onmouseover", "onmouseout", etc. (JavaScript event handlers)
        malicious_event_handlers = ['onmouseover', 'onmouseout', 'onmousedown', 'onmouseup', 'onmousemove', 'onkeydown', 'onkeyup', 'onkeypress']
        feature14 = sum(lower_script.count(handler) for handler in malicious_event_handlers)

        # Feature 15: Count of "javascript:" in attribute values
        feature15 = lower_script.count('javascript:')

        # Feature 16: Count of "data:" in attribute values
        feature16 = lower_script.count('data:')

        # Feature 17: Count of "expression(" (Internet Explorer-specific property)
        feature17 = lower_script.count('expression(')

        # Feature 18: Count of "<iframe" with "src" attribute
        feature18 = lower_script.count('<iframe') + lower_script.count('&lt;iframe') + lower_script.count('%3ciframe') + lower_script.count('%3c%69%66%72%61%6d%65')

        # Feature 19: Count of "alert" with various encodings
        feature19 = lower_script.count('alert') + lower_script.count('&lt;alert') + lower_script.count('%61%6c%65%72%74')

        ajax_keywords = ['xmlhttprequest', 'fetch', 'axios', '$.ajax', 'ajax','eval', 'document.cookie', 'innerHTML', 'createElement', 'setAttribute', 'onerror', 'onclick', 'onload',
                          'XMLHttpRequest', 'fetch', '$.ajax','unescape', 'decodeURIComponent', 'encodeURIComponent', 'escape','onmouseover', 'onerror', 'onload',
                         'document.write', 'document.writeln', 'element.innerHTML', 'element.outerHTML']
        feature20 = sum(lower_script.count(keyword) for keyword in ajax_keywords)

        # Append the new features to the feature vector
        #feature_vec = [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12, feature13, feature14, feature15, feature16, feature17, feature18, feature19]
        feature_vec = [feature1,feature6,feature7,feature14,feature18,feature19]
        features.append(feature_vec)

    return np.array(features)

features = extract_features_from_scripts(data['Sentence'])
data['features'] = list(features)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, data["Label"], test_size = .2, random_state=42)

#Creating a model for Random Forest,Gradient Boosting,Ada Boost,SVM
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC

random_state = 42

# Random Forest Classifier
my_rf_classifier = RandomForestClassifier(n_estimators=75, random_state=random_state)

# Gradient Boosting Classifier
my_gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=random_state)

# AdaBoost Classifier
my_ada_classifier = AdaBoostClassifier(n_estimators=100, random_state=random_state)

# Support Vector Machine (SVM) Classifier
my_svm_classifier = SVC(kernel='linear', random_state=random_state)

from sklearn.model_selection import train_test_split
# Training and predictions for Random Forest
my_rf_classifier.fit(X_train, y_train)
rf_predictions = my_rf_classifier.predict(X_test)

# Training and predictions for Gradient Boosting
my_gb_classifier.fit(X_train, y_train)
gb_predictions = my_gb_classifier.predict(X_test)

# Training and predictions for AdaBoost
my_ada_classifier.fit(X_train, y_train)
ada_predictions = my_ada_classifier.predict(X_test)

# Training and predictions for SVM
my_svm_classifier.fit(X_train, y_train)
svm_predictions = my_svm_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion matrix for Random Forest
rf_conf_matrix = confusion_matrix(y_test, rf_predictions)

# Confusion matrix for Gradient Boosting
gb_conf_matrix = confusion_matrix(y_test, gb_predictions)

# Confusion matrix for SVM
svm_conf_matrix = confusion_matrix(y_test, svm_predictions)

from sklearn.metrics import accuracy_score

# Calculate and print accuracy scores for each classifier
accuracy_rf = accuracy_score(y_test, rf_predictions)
accuracy_gb = accuracy_score(y_test, gb_predictions)
accuracy_ada = accuracy_score(y_test, ada_predictions)
accuracy_svm = accuracy_score(y_test, svm_predictions)

from sklearn.metrics import classification_report

my_rf_classifier.fit(features, data["Label"])

# print("Training Classifier 2 Gradient Boosting")
my_gb_classifier.fit(features, data["Label"])

my_ada_classifier.fit(features, data["Label"])

# print("Training Classifier 4 SVM")
my_svm_classifier.fit(features, data["Label"])

from urllib.parse import urlparse, parse_qs

def classify_urls(url):

    # Initialize return variables
    xss_yes = None
    non_malicious_url = ""

    # Extract the query part of the URL
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)

    # Assuming we're interested in the 'input' parameter for XSS checks
    xss_input = query_params.get('input', [''])[0]  # Get the 'input' value or default to an empty string if not present

    # If there's no 'input' to evaluate, consider this as non-malicious
    if not xss_input:
        non_malicious_url = url
        return {
            "xss_urls": None,  # No XSS if no input is provided
            "non_malicious_urls": non_malicious_url
        }

    # Extract features from the input
    Xnew1 = extract_features_from_scripts([xss_input])  # Pass the input as a list

    # Make predictions with each classifier
    ynew11 = my_rf_classifier.predict(Xnew1)
    ynew21 = my_gb_classifier.predict(Xnew1)
    ynew31 = my_ada_classifier.predict(Xnew1)
    ynew41 = my_svm_classifier.predict(Xnew1)

    # Calculate the score based on the classifiers' predictions
    score = (0.40 * ynew11[0] + 0.35 * ynew21[0] + 0.15 * ynew31[0] + 0.10 * ynew41[0])

    # Classify as XSS or NOT XSS
    if score >= 0.5:
        xss_yes = url  # Consider the URL as XSS if score is >= 0.5
    else:
        # Modify URL based on criteria for non-malicious URLs
        if "/non_critical/" in url:
            non_malicious_url = url.replace("/non_critical/", "?db_type=non_critical/")
        elif "/critical/" in url:
            non_malicious_url = url.replace("/critical/", "?db_type=critical/")
        else:
            non_malicious_url = url  # If no modifications needed, keep the original URL

    # Return the results
    return {
        "xss_urls": xss_yes,  # If XSS detected, the URL goes here
        "non_malicious_urls": non_malicious_url  # If no XSS, return the modified or original URL
    }

# Test the function with a sample URL
test_url = "http://localhost/?db_type=non_critical&input=<script>alert('xss')</script>"
non_malicious_results = classify_urls(test_url)
print(non_malicious_results)