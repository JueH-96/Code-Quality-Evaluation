# YOUR CODE HERE
def find_middle_brother(s_ab, s_ac, s_bc):
    ages = {'A': 0, 'B': 0, 'C': 0}
    
    if s_ab == '<':
        ages['B'] += 1
    else:
        ages['A'] += 1
    
    if s_ac == '<':
        ages['C'] += 1
    else:
        ages['A'] += 1
    
    if s_bc == '<':
        ages['C'] += 1
    else:
        ages['B'] += 1
    
    return sorted(ages.items(), key=lambda x: x[1])[1][0]

s_ab, s_ac, s_bc = input().split()
print(find_middle_brother(s_ab, s_ac, s_bc))