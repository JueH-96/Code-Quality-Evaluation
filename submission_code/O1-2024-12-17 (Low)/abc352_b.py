def main():
    import sys
    data = sys.stdin.read().strip().split()
    S, T = data[0], data[1]
    
    lenS = len(S)
    lenT = len(T)
    
    positions = []
    i = 0  # pointer for S
    j = 0  # pointer for T
    
    while i < lenS and j < lenT:
        if S[i] == T[j]:
            positions.append(j + 1)  # 1-based index
            i += 1
        j += 1
    
    print(" ".join(map(str, positions)))

if __name__ == "__main__":
    main()