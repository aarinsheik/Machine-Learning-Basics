#tut-4 : CodeBasics (ML)

# Gradient Descent and Cost Function : 

import numpy as np

def grad_des ( x , y ) :
    m_cur = b_cur = 0
    itr = 10000
    n = len(x) 
    learning_rate = 0.0001
    
    for i in range(itr) :
        
        y_pred = m_cur*x + b_cur

        cost = (1/n)*sum( [val**2 for val in ( y-y_pred) ] )
        
        md = -(2/n)*sum( x*(y - y_pred) )
        bd = -(2/n)*sum( y - y_pred ) 

        m_cur = m_cur  - learning_rate * md
        b_cur = b_cur - learning_rate * bd

        print(f"itr = {i+1} , m = {m_cur} , b = {b_cur} , cost = {cost}")


x = np.array([92,56,88,70,80,49,65,35,66,67])
y = np.array([98,68,81,80,83,52,66,30,68,73])

grad_des(x,y)