from typing import List

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float],
                  pairs2: List[List[str]], rates2: List[float]) -> float:
        # We'll use a Floyd Warshall–style approach because the graphs are small.
        # On each day, for every conversion provided, you also have the reverse conversion.
        # For day1, we want the maximum factor by which you can convert initialCurrency to any other currency.
        # For day2, we want the maximum factor by which you can convert from any currency back to initialCurrency.
        #
        # Then, the answer is:
        #    max_{currency X}  ( (conversion factor on day1 from initialCurrency to X) *
        #                         (conversion factor on day2 from X to initialCurrency) )
        #
        # First, build the set of all currencies available from both days.
        currencies = set()
        currencies.add(initialCurrency)
        for pair in pairs1:
            currencies.add(pair[0])
            currencies.add(pair[1])
        for pair in pairs2:
            currencies.add(pair[0])
            currencies.add(pair[1])
        currencies = list(currencies)
        n = len(currencies)
        
        # Map currency to index.
        idx = {c: i for i, c in enumerate(currencies)}
        
        # Build conversion matrix for day1.
        # dp1[i][j] will represent the maximum conversion factor from currency i to j on day1.
        dp1 = [[0.0] * n for _ in range(n)]
        for i in range(n):
            dp1[i][i] = 1.0
        for (pair, rate) in zip(pairs1, rates1):
            a, b = pair
            i_a, i_b = idx[a], idx[b]
            # given conversion a -> b at rate and b -> a at 1/rate.
            dp1[i_a][i_b] = max(dp1[i_a][i_b], rate)
            dp1[i_b][i_a] = max(dp1[i_b][i_a], 1.0 / rate)
            
        # Run Floyd Warshall style updates to allow multi-step conversions for day1.
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dp1[i][k] > 0 and dp1[k][j] > 0:
                        candidate = dp1[i][k] * dp1[k][j]
                        if candidate > dp1[i][j]:
                            dp1[i][j] = candidate
                            
        # Build the conversion matrix for day2.
        # Here, dp2[i][j] is the maximum conversion factor from currency i to j on day2.
        dp2 = [[0.0] * n for _ in range(n)]
        for i in range(n):
            dp2[i][i] = 1.0
        for (pair, rate) in zip(pairs2, rates2):
            a, b = pair
            i_a, i_b = idx[a], idx[b]
            dp2[i_a][i_b] = max(dp2[i_a][i_b], rate)
            dp2[i_b][i_a] = max(dp2[i_b][i_a], 1.0 / rate)
            
        # Run Floyd Warshall on day2 to allow multi-step conversions on day2.
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dp2[i][k] > 0 and dp2[k][j] > 0:
                        candidate = dp2[i][k] * dp2[k][j]
                        if candidate > dp2[i][j]:
                            dp2[i][j] = candidate
                            
        # Now, our plan is:
        #   (1) Convert 1 unit of the initial currency on day1 into some currency X,
        #   (2) Convert currency X back to the initial currency on day2.
        # The overall conversion factor for using X as an intermediate is:
        #         dp1[initial][X] * dp2[X][initial]
        # We then return the maximum such product over all X.
        init_index = idx[initialCurrency]
        ans = 0.0
        for x in range(n):
            candidate = dp1[init_index][x] * dp2[x][init_index]
            if candidate > ans:
                ans = candidate
        return ans

# Example usage and tests:
if __name__ == "__main__":
    sol = Solution()
    
    # Example 1:
    initialCurrency = "EUR"
    pairs1 = [["EUR", "USD"], ["USD", "JPY"]]
    rates1 = [2.0, 3.0]
    pairs2 = [["JPY", "USD"], ["USD", "CHF"], ["CHF", "EUR"]]
    rates2 = [4.0, 5.0, 6.0]
    # Day1: EUR -> USD (2.0), then USD -> JPY (3.0) gives 6.0 JPY.
    # Day2: JPY -> USD (4.0), USD -> CHF (5.0), CHF -> EUR (6.0) gives 720.0 EUR.
    print(sol.maxAmount(initialCurrency, pairs1, rates1, pairs2, rates2))  # Expected output: 720.0

    # Example 2:
    initialCurrency = "NGN"
    pairs1 = [["NGN", "EUR"]]
    rates1 = [9.0]
    pairs2 = [["NGN", "EUR"]]
    rates2 = [6.0]
    # Day1: NGN -> EUR: 9.0 EUR.
    # Day2: must convert back: EUR -> NGN using reverse conversion at rate 1/6, yielding 9/6 = 1.5.
    print(sol.maxAmount(initialCurrency, pairs1, rates1, pairs2, rates2))  # Expected output: 1.5

    # Example 3:
    initialCurrency = "USD"
    pairs1 = [["USD", "EUR"]]
    rates1 = [1.0]
    pairs2 = [["EUR", "JPY"]]
    rates2 = [10.0]
    # Here it is optimal to do no conversion since the return path from EUR on day2 is not helpful.
    print(sol.maxAmount(initialCurrency, pairs1, rates1, pairs2, rates2))  # Expected output: 1.0