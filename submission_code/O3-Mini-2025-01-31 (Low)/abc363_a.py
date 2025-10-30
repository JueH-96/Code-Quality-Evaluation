def main():
    import sys
    input = sys.stdin.readline
    R = int(input().strip())
    if R < 100:
        target = 100
    elif R < 200:
        target = 200
    elif R < 300:
        target = 300
    # Since R is <= 299, we always have a target
    print(target - R)

if __name__ == '__main__':
    main()