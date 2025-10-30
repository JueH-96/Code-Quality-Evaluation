This code looks correct implementation of the recurrence.  Why is it giving 0 for example 1?

Let's trace example 1: `prices = [3, 1, 2]`. n = 3.
`dp[4] = 0`
`dp[3] = min(prices[2] + dp[min(4, 7)], dp[4]) = min(2 + dp[4], dp[4]) = min(2 + 0, 0) = 0`.
`dp[2] = min(prices[1] + dp[min(4, 5)], dp[3]) = min(1 + dp[5], dp[3]) = min(1 + 0, 0) = 0`.
`dp[1] = min(prices[0] + dp[min(4, 3)], dp[2]) = min(3 + dp[3], dp[2]) = min(3 + 0, 0) = 0`.  Still 0.

Maybe the recurrence is wrong.  Let's re-examine the options for fruit `i`.
We must acquire fruit `i`.
Option 1: Buy fruit `i`. Cost `prices[i-1]`.  We get fruits `i+1, ..., i+i` free.  Then we need to acquire fruits from `i + i + 1` to `n`.
Option 2: Don't buy fruit `i`.  Then we must rely on getting fruit `i` for free from a previous purchase.  If we can get fruit `i` for free, we should take it if it's optimal.  But how to determine if we can get fruit `i` for free?  It depends on whether we bought some fruit `j < i` such that `i <= j + j`.

Maybe we should consider the fruits one by one. For each fruit, we decide whether to buy it or get it free.


## Part 1