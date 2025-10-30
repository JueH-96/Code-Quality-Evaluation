# YOUR CODE HERE
def count_disinfected_aliens(N, M, hands):
    count = 0
    remaining_disinfectant = M
    
    for h in hands:
        if remaining_disinfectant >= h:
            count += 1
            remaining_disinfectant -= h
        else:
            remaining_disinfectant = 0  # Use up all remaining disinfectant
        
    return count

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    hands = list(map(int, data[2:2 + N]))
    
    result = count_disinfected_aliens(N, M, hands)
    print(result)