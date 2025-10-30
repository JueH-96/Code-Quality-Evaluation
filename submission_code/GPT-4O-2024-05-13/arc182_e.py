# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    C = int(data[2])
    K = int(data[3])
    A = list(map(int, data[4:]))
    
    min_mods = [min((C * k + a) % M for a in A) for k in range(K)]
    result = sum(min_mods)
    
    print(result)

if __name__ == "__main__":
    main()