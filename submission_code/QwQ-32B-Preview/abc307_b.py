def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    strings = data[1:N+1]
    for i in range(N):
        for j in range(N):
            if i != j:
                concat = strings[i] + strings[j]
                if concat == concat[::-1]:
                    print("Yes")
                    return
    print("No")

if __name__ == '__main__':
    main()