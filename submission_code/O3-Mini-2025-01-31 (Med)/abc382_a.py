def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    D = int(input_data[1])
    S = input_data[2]
    
    # Count the number of boxes with cookies initially.
    initial_cookies = S.count('@')
    
    # Each day one cookie is eaten. Thus, D boxes which contained cookies become empty.
    # The empty boxes after D days are those that were originally empty plus the boxes where cookies were eaten.
    # Thus, answer = (number of originally empty boxes) + D = (N - initial_cookies) + D.
    answer = (N - initial_cookies) + D
    
    print(answer)

if __name__ == '__main__':
    main()