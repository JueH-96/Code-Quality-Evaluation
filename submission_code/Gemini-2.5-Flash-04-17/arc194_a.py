import heapq

def solve():
    N = int(input())
    A = list(map(int, input().split()))

    current_sum = 0
    pq = [] # min-priority queue to store elements currently considered to be on the stack

    for x in A:
        # Option 1: Tentatively append A_i to the sequence of elements we are keeping.
        # Add A_i to the current sum and push it onto the priority queue.
        # This represents using one operation (a push).
        current_sum += x
        heapq.heappush(pq, x)

        # The elements in the priority queue represent the set of values currently on the stack.
        # The sum of elements in the priority queue is `current_sum`.
        # If the current sum becomes negative, it means that the set of elements we have
        # decided to keep so far reduces the total sum. To maximize the sum, we should
        # reconsider the worst decision made so far, which is keeping the smallest (most negative) element.
        # We use a conceptual 'pop' operation to remove this smallest element.
        # This increases the sum by subtracting the negative value (adding the positive absolute value).
        # This conceptual pop operation uses one of the available operation slots.
        # The logic is that if the sum turns negative, we *must* use an operation to fix it.
        # The most beneficial operation is to remove the smallest element kept so far.
        # This implicitly means replacing a conceptual 'push' of the smallest element
        # with a 'pop' operation.

        # If the current sum is negative, remove the smallest element from the priority queue
        # and subtract its value from the sum. This corresponds to using a pop operation
        # to discard the element that contributes least positively (or most negatively)
        # to the sum of elements currently on the stack.
        if current_sum < 0:
            # We must have elements in pq if current_sum is negative after adding x
            # unless A_i itself is negative and pq was empty.
            # If pq is empty and current_sum is negative, it will remain negative.
            # The problem constraint says if S is empty, we *must* append.
            # If pq was empty before adding x, we must add x. If x is negative, sum becomes negative.
            # In this case, we cannot pop, so the sum remains negative.
            # If pq was not empty, and current_sum < 0, we can perform a pop conceptually.
            # The most beneficial element to pop is the smallest one currently contributing to the sum.
            if pq: # Ensure pq is not empty before popping
                removed_val = heapq.heappop(pq)
                current_sum -= removed_val

    # The final sum is the sum of the elements that remain in the priority queue.
    # These are the elements that constitute the sequence S after all operations,
    # selected to maximize the sum. The number of elements in the PQ is the final
    # stack size, which must be non-negative. The operations sequence implicitly
    # satisfies the stack properties and the total operation count.

    # The sum of elements in the PQ is the maximum sum.
    print(current_sum)

solve()