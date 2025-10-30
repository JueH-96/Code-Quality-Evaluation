import math

def get_freq(s):
    freq = [0] * 10
    for ch in s:
        freq[int(ch)] += 1
    return tuple(freq)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    freq_S = get_freq(S)
    max_num = 10**N - 1
    max_i = int(math.isqrt(max_num))
    count = 0
    for i in range(0, max_i + 1):
        square = i * i
        s_square = str(square)
        freq_square = get_freq(s_square)
        if freq_square == freq_S:
            count += 1
    print(count)

if __name__ == "__main__":
    main()