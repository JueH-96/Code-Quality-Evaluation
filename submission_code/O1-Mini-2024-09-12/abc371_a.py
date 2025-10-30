# YOUR CODE HERE
def main():
    S_AB, S_AC, S_BC = input().split()
    count_old = {'A':0, 'B':0, 'C':0}
    
    if S_AB == '<':
        count_old['A'] +=1
    else:
        count_old['B'] +=1
    
    if S_AC == '<':
        count_old['A'] +=1
    else:
        count_old['C'] +=1
    
    if S_BC == '<':
        count_old['B'] +=1
    else:
        count_old['C'] +=1
    
    for brother in ['A', 'B', 'C']:
        if count_old[brother] ==1:
            print(brother)
            break

if __name__ == "__main__":
    main()