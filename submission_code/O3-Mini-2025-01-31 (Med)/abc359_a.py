def main():
    import sys
    input_data = sys.stdin.read().splitlines()
    N = int(input_data[0])
    # Count how many times "Takahashi" appears in the list of strings from line 1 to N.
    count_takahashi = sum(1 for i in range(1, N+1) if input_data[i] == "Takahashi")
    print(count_takahashi)

if __name__ == '__main__':
    main()