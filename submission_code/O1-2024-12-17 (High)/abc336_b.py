def main():
    import sys
    
    N = int(sys.stdin.readline().strip())
    count = 0
    while N % 2 == 0:
        N //= 2
        count += 1
    print(count)

# Do not forget to call main
main()