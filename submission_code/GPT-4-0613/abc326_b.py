def main():
    N = int(input().strip())
    for i in range(N, 1000):
        hundreds = i // 100
        tens = (i // 10) % 10
        ones = i % 10
        if hundreds * tens == ones:
            print(i)
            break

main()