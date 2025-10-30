def approximate_n(N):
    N_str = str(N)
    D = len(N_str)
    if D <= 3:
        print(N)
    else:
        truncated = N_str[:3] + '0' * (D - 3)
        print(int(truncated))

if __name__ == "__main__":
    N = int(input())
    approximate_n(N)