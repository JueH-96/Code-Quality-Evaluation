def main():
    import sys
    data = sys.stdin.read().split()
    if data:
        N = int(data[0])
        L = int(data[1])
        scores = list(map(int, data[2:]))
        count_pass = sum(1 for score in scores if score >= L)
        print(count_pass)

if __name__ == "__main__":
    main()