import sys
import heapq

def main():
    """
    Solves the problem using a greedy approach with a min-heap.

    The problem requires performing N operations, one for each A_i. The operation
    is either appending A_i to a sequence S or deleting the last element of S.
    The goal is to maximize the sum of elements in S after all N operations.

    Let's analyze the constraints on the operations. After i steps (for A_1 to A_i),
    let p_i be the number of push operations and d_i be the number of pop operations.
    - Total operations: p_i + d_i = i
    - The stack size must remain non-negative, which implies that at any step t <= i,
      the number of pushes is at least the number of pops. This must hold for t=i, so p_i >= d_i.

    Combining these, we get p_i >= i - p_i, which simplifies to 2*p_i >= i, or p_i >= ceil(i/2).
    This means after processing the first i elements, we must have performed at least
    ceil(i/2) push operations to maintain a valid sequence of operations.

    To maximize the final sum, the elements we choose to push and keep should be the
    largest ones. This suggests a greedy strategy: at each step i, we maintain a
    collection of the largest numbers seen so far, with a size equal to the minimum
    number of pushes required.

    Algorithm:
    1. Initialize an empty min-heap `heap` and a `current_sum` to 0.
    2. Iterate through the input array `a` with index `i` from 0 to N-1.
       Let the current element be `x = a[i]`.
    3. Add `x` to the `heap` and to the `current_sum`.
    4. The number of operations performed is `i+1`. The minimum required number of
       pushes is `k = ceil((i+1)/2)`. In integer arithmetic, this is `(i + 1 + 1) // 2` or `(i + 2) // 2`.
    5. If the number of elements in our heap `len(heap)` is greater than `k`, it means
       we have one more "push" than minimally required. To maximize the sum, we should
       get rid of the smallest element among the ones we've pushed. This is achieved by
       popping from the min-heap.
    6. We remove the smallest element from the heap and subtract it from `current_sum`.
    7. After the loop, `current_sum` holds the maximum possible sum of elements in S.
    """
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))

    heap = []
    current_sum = 0

    for i, x in enumerate(a):
        # Add the current element to our set of pushed items.
        heapq.heappush(heap, x)
        current_sum += x

        # The number of operations so far is i+1.
        # The minimum number of pushes required is ceil((i+1)/2).
        # In 0-indexed integer arithmetic, this is (i+2)//2.
        required_pushes = (i + 2) // 2

        # If we have more elements than the minimum required pushes,
        # it means we can "convert" a push of the smallest element into a pop.
        # This is done by removing the smallest element from our collection.
        if len(heap) > required_pushes:
            smallest = heapq.heappop(heap)
            current_sum -= smallest

    print(current_sum)

if __name__ == "__main__":
    main()