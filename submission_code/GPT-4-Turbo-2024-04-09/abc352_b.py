import sys
input = sys.stdin.read

def main():
    data = input().split()
    S = data[0]
    T = data[1]
    
    positions = []
    index_s = 0
    index_t = 0
    
    while index_s < len(S) and index_t < len(T):
        if S[index_s] == T[index_t]:
            positions.append(index_t + 1)  # 1-based index
            index_s += 1
        index_t += 1
    
    print(" ".join(map(str, positions)))

if __name__ == "__main__":
    main()