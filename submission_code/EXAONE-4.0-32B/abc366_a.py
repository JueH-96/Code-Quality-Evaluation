def main():
    import sys
    data = sys.stdin.readline().split()
    n = int(data[0])
    t = int(data[1])
    a = int(data[2])
    r = n - t - a
    
    if t > a + r or a > t + r:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()