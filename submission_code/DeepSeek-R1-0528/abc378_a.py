def main():
    data = list(map(int, input().split()))
    freq = [0] * 5
    for num in data:
        freq[num] += 1
    total_ops = freq[1] // 2 + freq[2] // 2 + freq[3] // 2 + freq[4] // 2
    print(total_ops)

if __name__ == "__main__":
    main()