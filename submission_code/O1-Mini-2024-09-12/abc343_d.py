# YOUR CODE HERE
import sys

def main():
    import sys
    from sys import stdin
    import sys
    import sys
    N, T = map(int, stdin.readline().split())
    scores = [0] * (N+1)  # 1-based index
    freq = {}
    freq[0] = N
    distinct = 1
    output = []
    for _ in range(T):
        A_i, B_i = map(int, stdin.readline().split())
        s = scores[A_i]
        freq[s] -= 1
        if freq[s] == 0:
            distinct -= 1
            del freq[s]
        s_new = s + B_i
        scores[A_i] = s_new
        if s_new in freq:
            freq[s_new] += 1
        else:
            freq[s_new] = 1
            distinct += 1
        output.append(str(distinct))
    print('
'.join(output))

if __name__ == "__main__":
    main()