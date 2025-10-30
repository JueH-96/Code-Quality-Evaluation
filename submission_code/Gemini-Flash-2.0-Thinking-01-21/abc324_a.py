# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    if n == 0:
        print("Yes") # Consider edge case of empty input, though problem states N >= 2. But just in case. If no numbers, are they all equal? Let's assume yes.
    elif n == 1:
        print("Yes") # If only one number, it's trivially equal to itself.
    else:
        first_element = a[0]
        all_equal = True
        for i in range(1, n):
            if a[i] != first_element:
                all_equal = False
                break
        if all_equal:
            print("Yes")
        else:
            print("No")

if __name__ == '__main__':
    solve()