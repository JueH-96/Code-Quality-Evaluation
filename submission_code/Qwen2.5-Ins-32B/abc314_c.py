import sys

def shift_string_by_color(N, M, S, C):
    color_positions = [[] for _ in range(M)]
    for i in range(N):
        color_positions[C[i] - 1].append(i)
    
    S_list = list(S)
    for color in range(M):
        positions = color_positions[color]
        if positions:
            last_char = S_list[positions[-1]]
            for i in range(len(positions) - 1, 0, -1):
                S_list[positions[i]] = S_list[positions[i - 1]]
            S_list[positions[0]] = last_char
    
    return ''.join(S_list)

if __name__ == "__main__":
    input = sys.stdin.read
    N, M, S, *C = input().split()
    N, M = int(N), int(M)
    C = list(map(int, C))
    result = shift_string_by_color(N, M, S, C)
    print(result)