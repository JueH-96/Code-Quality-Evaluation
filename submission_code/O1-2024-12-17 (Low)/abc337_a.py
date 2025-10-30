def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])

    sum_taka = 0
    sum_aoki = 0
    index = 1

    for _ in range(N):
        x = int(data[index])
        y = int(data[index+1])
        sum_taka += x
        sum_aoki += y
        index += 2

    if sum_taka > sum_aoki:
        print("Takahashi")
    elif sum_aoki > sum_taka:
        print("Aoki")
    else:
        print("Draw")

# Do not remove the below call to main()
if __name__ == "__main__":
    main()