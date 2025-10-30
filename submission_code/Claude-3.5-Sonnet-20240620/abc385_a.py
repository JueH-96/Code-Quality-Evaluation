# YOUR CODE HERE
def can_divide_equally(a, b, c):
    total = a + b + c
    
    # Check if the total can be divided into 2 equal parts
    if total % 2 == 0:
        half = total // 2
        if a == half or b == half or c == half or a + b == half or b + c == half or a + c == half:
            return True
    
    # Check if the total can be divided into 3 equal parts
    if total % 3 == 0:
        third = total // 3
        if (a == third and b == third) or (a == third and c == third) or (b == third and c == third):
            return True
    
    return False

# Read input
a, b, c = map(int, input().split())

# Check if it's possible to divide equally and print the result
if can_divide_equally(a, b, c):
    print("Yes")
else:
    print("No")