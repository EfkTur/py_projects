import os 

def total_detection_counter():
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