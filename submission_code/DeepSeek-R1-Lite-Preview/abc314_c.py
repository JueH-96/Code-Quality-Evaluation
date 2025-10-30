import sys

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    S = data[2]
    C = list(map(int, data[3:3+N]))
    
    s_list = list(S)
    positions = [[] for _ in range(M)]
    
    for i in range(N):
        color = C[i] - 1  # Convert to 0-based index
        positions[color].append(i)
    
    for color in range(M):
        pos_list = positions[color]
        chars = [s_list[p] for p in pos_list]
        shifted_chars = [chars[-1]] + chars[:-1]
        for p, c in zip(pos_list, shifted_chars):
            s_list[p] = c
    
    S_final = ''.join(s_list)
    print(S_final)

if __name__ == "__main__":
    main()