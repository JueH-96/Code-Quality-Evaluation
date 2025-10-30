# YOUR CODE HERE
import sys

def get_tile(x, y, K):
    i = (2 * x + 1) // (2 * K)
    j = (2 * y + 1) // (2 * K)
    if (i & 1) == (j & 1):
        k = y - j * K
    else:
        k = x - i * K
    return (i, j, k)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    T = int(data[0])
    idx =1
    results=[]
    for _ in range(T):
        K = int(data[idx])
        S_x = int(data[idx+1])
        S_y = int(data[idx+2])
        T_x = int(data[idx+3])
        T_y = int(data[idx+4])
        idx +=5
        si, sj, sk = get_tile(S_x, S_y, K)
        ti, tj, tk = get_tile(T_x, T_y, K)
        delta_i = abs(ti - si)
        delta_j = abs(tj - sj)
        delta_k = abs(tk - sk)
        M = max(delta_i, delta_j, delta_k)
        answer = 2 * M
        results.append(str(answer))
    print('
'.join(results))

if __name__ == "__main__":
    main()