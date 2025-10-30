def main():
    import sys
    from collections import Counter

    s = sys.stdin.readline().strip()
    cnt = Counter(s)
    if cnt.get('1', 0) == 1 and cnt.get('2', 0) == 2 and cnt.get('3', 0) == 3:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()