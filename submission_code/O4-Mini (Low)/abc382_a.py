def main():
    import sys
    input = sys.stdin.readline
    
    N, D = map(int, input().split())
    S = input().rstrip()
    
    total_cookies = S.count('@')
    remaining_cookies = total_cookies - D
    empty_boxes = N - remaining_cookies
    
    print(empty_boxes)

if __name__ == "__main__":
    main()