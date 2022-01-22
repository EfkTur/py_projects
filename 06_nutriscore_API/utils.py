def grader_food(x):
        if x<=-1:
            return 'a'
        elif (x>-1)&(x<=2):
            return 'b'
        elif (x>2)&(x<=10):
            return 'c'
        elif (x>10)&(x<=18):
            return 'd'
        else:
            return 'e'

def grader_beverages(x):
        if (x<=0):
            return 'a'
        elif (x>0)&(x<=1):
            return 'b'
        elif (x>1)&(x<=5):
            return 'c'
        elif (x>5)&(x<=9):
            return 'd'
        else:
            return 'e'
