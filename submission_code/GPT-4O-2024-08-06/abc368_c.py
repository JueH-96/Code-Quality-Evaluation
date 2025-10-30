# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    H = list(map(int, data[1:]))
    
    T = 0
    current_enemy = 0
    
    while current_enemy < N:
        T += 1
        if T % 3 == 0:
            H[current_enemy] -= 3
        else:
            H[current_enemy] -= 1
        
        if H[current_enemy] <= 0:
            current_enemy += 1
    
    print(T)

if __name__ == "__main__":
    main()