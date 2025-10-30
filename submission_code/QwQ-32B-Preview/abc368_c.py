def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    H = list(map(int, data[1:N+1]))
    K = [((h // 5) * 3) + (h % 5 if h % 5 < 3 else 3) for h in H]
    total_T = sum(K)
    print(total_T)

if __name__ == "__main__":
    main()