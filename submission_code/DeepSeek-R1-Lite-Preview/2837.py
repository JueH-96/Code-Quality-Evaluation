class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        def can_partition(S, k):
            if S < k:
                return False
            # Start with the highest power of 2 less than or equal to S
            power = 60
            remaining = S
            count = 0
            while power >= 0 and count < k and remaining > 0:
                if remaining >= (1 << power):
                    remaining -= (1 << power)
                    count += 1
                else:
                    power -= 1
            # If we have used exactly k operations and remaining is zero
            if count == k and remaining == 0:
                return True
            # If we have used fewer than k operations, check if remaining can be split into (k - count) parts of 1
            if remaining <= k - count:
                return True
            return False
        
        # Binary search for the minimal k
        left = 1
        # Set an upper bound for k
        if num2 < 0:
            # As k increases, S increases. To prevent infinite loop, set a reasonable upper limit
            right = (10**14) // abs(num2) + 100
        else:
            right = num1
        answer = -1
        while left <= right:
            mid = (left + right) // 2
            S = num1 - mid * num2
            if S < mid:
                # Not possible for this k
                left = mid + 1
                continue
            if can_partition(S, mid):
                answer = mid
                right = mid - 1
            else:
                left = mid + 1
        return answer