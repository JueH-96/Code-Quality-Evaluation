def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    D = int(data[1])
    S = data[2]
    
    # Count the number of cookies initially present.
    num_cookies = S.count('@')
    # Initially, there are (N - num_cookies) empty boxes.
    # Each day Takahashi eats one cookie from a box that initially had a cookie,
    # making that box empty. Since he eats D cookies over D days,
    # a total of D additional boxes will be empty.
    result = (N - num_cookies) + D
    
    sys.stdout.write(str(result))
    
if __name__ == '__main__':
    main()