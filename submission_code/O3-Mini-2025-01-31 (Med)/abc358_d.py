def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    # Read the candy counts/prices of the boxes
    A = [int(next(it)) for _ in range(n)]
    # Read the candy requirements for each person
    B = [int(next(it)) for _ in range(m)]
    
    # To achieve the minimum cost, we want to assign to each person the cheapest available box that meets their requirement.
    # Since the cost is equal to the candy count, we can sort both lists in non-decreasing order.
    A.sort()
    B.sort()
    
    total_cost = 0
    box_index = 0
    
    # For each person's requirement in increasing order, pick the earliest available box in A that is large enough.
    for requirement in B:
        # Skip boxes that don't satisfy the current requirement.
        while box_index < n and A[box_index] < requirement:
            box_index += 1
        # If we've run out of boxes, it's impossible to serve this person.
        if box_index >= n:
            sys.stdout.write("-1")
            return
        # Use the current box, add its cost, and then move to the next box.
        total_cost += A[box_index]
        box_index += 1
        
    sys.stdout.write(str(total_cost))

if __name__ == '__main__':
    main()