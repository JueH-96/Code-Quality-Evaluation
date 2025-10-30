class Solution:
    """
    Solves the problem of finding the minimum `ans[i]` such that 
    `ans[i] | (ans[i] + 1) == nums[i]` for each prime `nums[i]`.
    """
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        """
        Constructs the array `ans` based on the given prime array `nums`.

        Args:
            nums: A list of prime integers.

        Returns:
            A list `ans` where `ans[i]` is the minimum non-negative integer
            satisfying `ans[i] | (ans[i] + 1) == nums[i]`, or -1 if no such
            integer exists.
        """
        ans = []
        for num in nums:
            # Case 1: num is 2.
            # We need to find x such that x | (x + 1) == 2.
            # Let's test small non-negative x:
            # x=0: 0 | (0 + 1) = 0 | 1 = 1 != 2
            # x=1: 1 | (1 + 1) = 1 | 2 = 3 != 2
            # If x is even (x >= 2), let x = 2k (k >= 1). Then x | (x + 1) = x + 1.
            # We need x + 1 = 2, which means x = 1. But x must be even, a contradiction.
            # If x is odd (x >= 1), let x = 2k + 1. Let x end in `k'` ones (k' >= 1).
            # Then x | (x + 1) will end in `k' + 1` ones. 
            # Since k' >= 1, k' + 1 >= 2. So x | (x + 1) must end in at least two ones (...11).
            # The number 2 is 10 in binary, which does not end in two ones.
            # Therefore, no non-negative integer x satisfies the condition for num = 2.
            if num == 2:
                ans.append(-1)
                continue

            # Case 2: num is an odd prime (num >= 3).
            # We need to find the minimum non-negative integer x such that x | (x + 1) == num.
            
            # There are two potential types of solutions for x: even or odd.
            
            # Possibility A: x is even.
            # If x is even, then x | (x + 1) = x + 1. (Proven by checking binary representations)
            # For this to equal num, we must have x + 1 = num, which implies x = num - 1.
            # Since num is an odd prime >= 3, num - 1 is an even integer >= 2.
            # Thus, x_even = num - 1 is always a valid candidate solution.
            x_even = num - 1
            
            # Possibility B: x is odd.
            # If x is odd, let x end in k ones (k >= 1). `x = ...01...1` (k ones).
            # Then `x + 1 = ...10...0` (k zeros).
            # And `y = x | (x + 1) = ...11...1` (k+1 ones), where the leading bits are determined by the part of x before the trailing ones.
            # Specifically, if x = A'0(1^k), then y = A'1(1^k).
            # This structure implies that if x is odd, the result y = x | (x+1) must end 
            # in q = k+1 >= 2 consecutive ones in its binary representation.
            # Checking if a number ends in at least two ones is equivalent to checking `y % 4 == 3`.
            
            # Therefore, an odd solution x_odd can exist only if num % 4 == 3.
            
            if num % 4 == 3:
                # If num % 4 == 3, an odd solution x_odd exists.
                # We need to find it and determine if it's smaller than x_even.
                
                # Let q be the number of trailing ones in num's binary representation.
                # Since num % 4 == 3, we know q >= 2.
                
                # The number of trailing ones q can be found efficiently using the formula:
                # q = (num ^ (num + 1)).bit_length() - 1
                # Explanation: If num = ...z0(1^q), then num+1 = ...z1(0^q).
                # Their bitwise XOR is 0...01(1^q) = 2^(q+1) - 1 ? No.
                # XOR = 0...0 1 (0^q xor 1^q = 1^q). The total length of trailing ones is q.
                # Let's recheck: num=11 (1011), num+1=12 (1100). XOR = 0111. bit_length=3. q=3-1=2. OK.
                # num=7 (111), num+1=8 (1000). XOR = 1111. bit_length=4. q=4-1=3. OK.
                q = (num ^ (num + 1)).bit_length() - 1
                
                # We derived that the odd solution x_odd is given by num - 2^(q-1).
                # Calculate 2^(q-1) efficiently using left shift (1 << (q - 1)).
                power_of_2_q_minus_1 = 1 << (q - 1)
                x_odd = num - power_of_2_q_minus_1
                
                # Now we compare x_even = num - 1 and x_odd = num - 2^(q-1).
                # Since num % 4 == 3, we have q >= 2. This means q - 1 >= 1.
                # So, 2^(q-1) >= 2^1 = 2.
                # Because 2^(q-1) >= 2, it follows that 2^(q-1) > 1.
                # Multiplying by -1 reverses the inequality: -2^(q-1) < -1.
                # Adding num to both sides: num - 2^(q-1) < num - 1.
                # This shows that x_odd < x_even.
                
                # Since x_odd is smaller than x_even, x_odd is the minimum solution.
                ans.append(x_odd)
            else:
                # If num % 4 != 3 (which means num % 4 == 1, since num is an odd prime),
                # then no odd solution exists.
                # The only possible solution is the even one, x_even = num - 1.
                # Therefore, x_even must be the minimum solution.
                ans.append(x_even)
                
        return ans