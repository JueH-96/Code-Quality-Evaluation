def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    
    result = []
    for i in range(N + 1):
        char = '-'  # default if no condition is met
        for j in range(1, 10):  # j must be in 1..9
            if N % j == 0:  # j divides N
                step = N // j  # this is an integer because j divides N
                if i % step == 0:
                    char = str(j)
                    break  # choose the smallest such j and break
        result.append(char)
    
    sys.stdout.write("".join(result))

if __name__ == '__main__':
    main()