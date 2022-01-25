import os
from sklearn.dummy import DummyClassifier 

def total_detection_counter():
    """
    Raison d'être:
    This function that does return the number of detection in a given folder 
    which contains either training set or test set

    Args: 
    None

    Returns:
    None. Just prints the numbers of truth detections in the folder
    """

    num_lines = 0
    for filename in os.listdir('.'):
        # Checking if filename ends with '.txt'
        if filename.endswith(".txt"):
            sub_file = open(filename,"r")
            num_lines += sum(1 for line in sub_file)
            sub_file.close()
    print(num_lines)

total_detection_counter()

def total_masked_counter():
    """
    Raison d'être:
    This function that does return the number of truth masked detection in a given folder 
    which contains either training set or test set
    
    Args:
    None

    Returns:
    None. Just prints the numbers of truth masked detections in the folder
    
    """

    num_line_sec = 0
    for filename in os.listdir('.'):
        # Checking if filename ends with '.txt'
        if filename.endswith(".txt"):
            sub_file = open(filename,"r")
            for line in sub_file:
                if line[0] == '0':
                    num_line_sec +=1
            sub_file.close()
    print(num_line_sec)

total_masked_counter()

def get_list_detections():
"""
    Raison d'être:
    This function is built to determine the class of each truth detection in the dataset
    
    Args:
    None

    Returns:
    List of truth detections ranked by class 
"""
    num_line_sec = 0
    results = []
    for filename in os.listdir('.'):
        # Checking if filename ends with '.txt'
        if filename.endswith(".txt"):
            sub_file = open(filename,"r")
            for line in sub_file:
                results.append(line[0])
            sub_file.close()
    print(results)

get_list_detections()

def dummy_counter():
    """



    """
    dummy = DummyClassifier(strategy='most_frequent')


    