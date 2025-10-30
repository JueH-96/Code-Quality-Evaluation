<think>
Okay, let's see. So the problem is about determining if all of Takahashi's plans can fall on holidays, given that the week structure is A holidays followed by B weekdays. The key is that the day of the week when each plan occurs must be a holiday. But since he doesn't know the current day, we have to check if there's any possible starting day that makes all D_i days later fall on holidays.

Hmm. Let's think about how the days cycle. The week has A+B days total. So each day after some days will repeat modulo (A+B). For each plan D_i, the day of the week it's on is (current_day + D_i) mod (A+B). If that mod result is between 1 and A, inclusive, then it's a holiday. So for all D_i, there must exist a starting day (let's call it X) such that for every D_i, (X + D_i) mod (A+B) is in [1, A].

But how do we find such an X? Since X can be any value from 1 to A+B (since the week cycles), but we need to check if there's an X that satisfies all D_i constraints.

Wait, but the problem says that today could be any day of the week. So X is the current day, which is between 1 and A+B. So for each D_i, the required condition is (X + D_i) mod (A+B) ≤ A. Wait, because the holidays are days 1 to A. So the mod result must be in 1..A. So for each D_i, X must be in ( (1 - D_i) mod (A+B) ) to ( (A - D_i) mod (A+B) ), but considering the wrap-around.

Alternatively, for each D_i, the allowed X values are those that satisfy X ≡ (1 - D_i) mod (A+B), (2 - D_i) mod (A+B), ..., (A - D_i) mod (A+B). So for each D_i, X must be in the set { (k - D_i) mod (A+B) | k = 1..A }.

But the problem is that for all D_i, their allowed X sets must have a non-empty intersection. If the intersection of all these sets is non-empty, then the answer is Yes. Otherwise, No.

So the approach is to compute for each D_i the set of possible X values that would make it a holiday, then check if the intersection of all these sets is non-empty.

But with N up to 2e5, we can't compute each set explicitly. So we need a smarter way to compute the intersection.

Wait, but each set for D_i is a range of X values modulo (A+B). Let's think about this. Let's denote S = A+B. For each D_i, the allowed X must satisfy (X + D_i) mod S <= A. Let's rephrase this:

(X + D_i) mod S <= A --> which is equivalent to X mod S <= (A - D_i) mod S, but wait, maybe not. Let's think again.

Wait, (X + D_i) mod S gives a value between 0 and S-1. But the holidays are 1 to A. So the condition is that (X + D_i) mod S is in [1, A]. Let's rephrase:

Let’s denote the day after D_i days as (X + D_i) mod S. If S is the total days in a week. But since the days are 1-based? Or 0-based?

Wait, the problem statement says the first through A-th days are holidays. So the days are numbered 1 to A, then A+1 to A+B. So the total days in a week is S = A+B. So the days are 1-based.

So when we compute (X + D_i) mod S, if the result is 0, that would correspond to day S (since 0 mod S is 0, but in 1-based terms, that's S). So for example, if X is S, then adding 1 day would be X+1 mod S = 1, which is a holiday.

So the condition for a plan D_i is that (X + D_i) mod S must be in [1, A]. So:

1 ≤ (X + D_i) mod S ≤ A.

Let’s rearrange this. Let’s let Y = X mod S. So Y is in [1, S]. Wait, but X can be any day from 1 to S. So Y = X, since X is already in 1..S.

Wait, perhaps the mod is a bit confusing here. Let's think of X as being in 1..S. Then (X + D_i) can be any integer. The mod S operation would give a value from 0 to S-1, but since we need to map to 1..S, perhaps the correct way is:

(X + D_i - 1) mod S + 1. Because, for example, if X is S, D_i is 1: (S +1 -1) mod S +1 = S mod S +1 = 0 +1 = 1. Which is correct.

But perhaps the problem's definition of mod is such that (X + D_i) mod S gives a number in 0..S-1, and then the day is (mod result) +1. Wait, that would make the days 1-based. So the actual day is ((X + D_i -1) mod S) + 1.

Alternatively, perhaps the problem considers that (X + D_i) mod S gives a number in 0..S-1, and then the day is that number +1. So:

day = (X + D_i) mod S
if day == 0: day = S

So the day is in 1..S. Then the condition is day ≤ A.

So for each D_i, the condition is (X + D_i) mod S <= A. But when the mod is 0, it's equivalent to S, which is only <= A if S <= A. But S = A+B, which is greater than A (since B >=1). So S <= A is impossible. So the mod 0 case would correspond to day S, which is a weekday. So the condition (X + D_i) mod S <= A can be rewritten as:

(X + D_i) mod S <= A.

But (X + D_i) mod S can be 0, which is not <= A (since A >=1, S = A+B >= A+1, so 0 < A is possible only if A >=1, but 0 is not <= A in that case. Wait, 0 is not in the range 1..A. So the mod must be in 1..A.

Wait, maybe I'm getting confused here. Let's re-express the condition.

The day after D_i days is (X + D_i) days later. Since the week cycles every S days, the actual day of the week is ((X + D_i -1) mod S) + 1. So:

If X is the current day (1-based), then adding D_i days gives X + D_i. Subtract 1 to make it 0-based, mod S, then add 1 to get back to 1-based.

So:

day = ((X + D_i -1) % S) + 1

We need day <= A.

So:

((X + D_i -1) % S) + 1 <= A

Subtract 1 from both sides:

(X + D_i -1) % S <= A -1

The mod S gives a value between 0 and S-1. So the condition is that (X + D_i -1) mod S is in [0, A-1].

Which is equivalent to (X + D_i -1) mod S <= A-1.

But (X + D_i -1) mod S is the same as (X + D_i -1) - k*S for some integer k, such that the result is in [0, S-1].

So the inequality is:

(X + D_i -1) mod S <= A-1

Which can be rewritten as:

(X + D_i -1) mod S ∈ [0, A-1]

So for each D_i, the allowed X values are those that satisfy this condition.

But X is in 1..S. So let's find the range of X that satisfies this for a given D_i.

Let’s rearrange the inequality:

Let’s denote X as a variable in [1, S]. Let’s compute (X + D_i -1) mod S. Let’s call this value M. So M must be <= A-1.

But M is (X + D_i -1) mod S. So:

X + D_i -1 ≡ M (mod S)

But M is between 0 and A-1. So:

X ≡ (M - D_i +1) mod S

But since X is in 1..S, we can write X = (M - D_i +1) mod S. Wait, perhaps not. Let me think.

Wait, X + D_i -1 ≡ M (mod S)

So X ≡ (M - D_i +1) mod S.

But M can be 0 to A-1. So for each possible M in 0..A-1, X can be (M - D_i +1) mod S. But since X must be in 1..S, the mod S will give a value in 0..S-1, so adding 1 if necessary.

Wait, this might not be the right approach. Let's think of X as a variable. For each D_i, the allowed X's are those such that when you add D_i-1 to X, the sum modulo S is in 0..A-1.

So (X + D_i -1) mod S is between 0 and A-1.

Which can be written as:

0 ≤ (X + D_i -1) mod S ≤ A-1

But (X + D_i -1) can be written as q*S + r, where r is between 0 and S-1. So the mod S is r. So the condition is r <= A-1.

So X + D_i -1 >= q*S + 0 and X + D_i -1 <= q*S + (A-1).

But X is in 1..S. Let's substitute X into the inequality.

Let’s find all X in [1, S] such that (X + D_i -1) mod S <= A-1.

Alternatively, (X + D_i -1) mod S is the same as (X + D_i -1 - k*S) where k is the integer such that the result is in [0, S-1].

So for X in [1, S], X + D_i -1 can range from (1 + D_i -1) = D_i to (S + D_i -1) = D_i + S -1.

So (X + D_i -1) mod S = (X + D_i -1) - S * floor( (X + D_i -1)/S )

But since X is up to S, X + D_i -1 can be up to S + D_i -1. But D_i can be up to 1e9. So this approach might not be helpful.

Alternatively, the condition (X + D_i -1) mod S <= A-1 is equivalent to X + D_i -1 ≡ r (mod S), where r is in [0, A-1].

Which can be rewritten as X ≡ (r - D_i +1) mod S.

So for each D_i, the allowed X's are the ones congruent to (r - D_i +1) mod S, for some r in 0..A-1.

But since r can be 0..A-1, the possible X's for D_i are the union of all residues (r - D_i +1) mod S for r in 0..A-1.

But this is the same as X ≡ ( (r - D_i +1) mod S ) for each r in 0..A-1.

But this is equivalent to X ≡ ( ( -D_i +1 + r ) mod S ) for r in 0..A-1.

So the possible X's for D_i are the residues ( ( -D_i +1 ) mod S ) + r, mod S, for r from 0 to A-1.

Wait, let's see. Let's take r from 0 to A-1:

For each r, (r - D_i +1) mod S = ( ( -D_i +1 + r ) mod S )

But since r ranges from 0 to A-1, the sum (-D_i +1) + r ranges from (-D_i +1) to (-D_i +1 + A-1) = (-D_i + A).

So the possible X residues are the values from (-D_i +1) mod S to (-D_i + A) mod S.

But since S is the modulus, this could wrap around.

So the allowed X's for D_i form a contiguous interval modulo S. For example, if (-D_i +1) mod S is s, then the allowed X's are s, s+1, ..., s + (A-1), mod S. But if s + (A-1) exceeds S, then the interval wraps around.

So for each D_i, the allowed X's form a range of A consecutive numbers modulo S. So the allowed X's for D_i can be represented as a range [L_i, R_i], where L_i = (-D_i +1) mod S, and R_i = (L_i + A -1) mod S. But since the range is over modulo S, we need to check if the interval is wrapped around.

So for example, if L_i <= R_i, then the allowed X's are all X in [L_i, R_i]. If L_i > R_i, then the allowed X's are [L_i, S] and [1, R_i].

But how do we represent this for each D_i and then find the intersection of all such ranges?

The key insight is that the allowed X's for each D_i form a contiguous interval (possibly wrapping around the modulus S). The intersection of all such intervals must be non-empty.

So the problem reduces to checking if the intersection of N intervals (each being a contiguous interval modulo S) is non-empty.

But how to compute this efficiently?

Let's consider that each interval is represented as a start and end, and whether it wraps around. For example, for D_i, the interval is [a, b], where a = (-D_i +1) mod S, and b = (a + A -1) mod S. Then, the interval is:

if a <= b: X must be in [a, b]

else: X must be in [a, S] or [1, b]

So for each D_i, the allowed X's are the union of [a, S] and [1, b] if a > b.

But how do we compute the intersection of all these intervals?

The approach is to find the minimal range that is covered by all intervals. For each interval, the allowed X's are a certain range. The intersection of all these ranges must be non-empty.

But with N up to 2e5, we need an O(N) approach.

Let me think. Let's represent the allowed X's for each D_i as a range [low_i, high_i], which may wrap around. Then, the intersection of all these ranges must be non-empty.

The idea is to compute the maximum of all low_i and the minimum of all high_i. If the maximum low_i <= the minimum high_i, then there's a common interval. Otherwise, there's no intersection.

But this approach works only if all the ranges are non-wrapping. If some ranges wrap around, this approach may not work.

Wait, but how to model the ranges that wrap around?

Alternatively, for each D_i, the allowed X's can be represented as two intervals: [a, S] and [1, b] if a > b. But handling this for all D_i would be complicated.

Alternatively, we can model the allowed X's for each D_i as a single interval in the circular space of S days. The intersection of all intervals is the set of X that are covered by all intervals. To find if this intersection is non-empty, we can track the maximum of the left endpoints and the minimum of the right endpoints, but considering the circular nature.

Wait, perhaps the correct approach is to represent each interval as a start and end, and compute the intersection as follows:

For each D_i, the allowed X's are [L_i, R_i], where L_i = (-D_i +1) mod S, R_i = (L_i + A -1) mod S.

But since the week is cyclic, we can represent each interval as a pair (L_i, R_i), and for the intersection, we need to find an X that is in all intervals.

But how?

An alternative approach is to find the minimal upper bound and maximal lower bound.

Wait, here's an idea from interval intersection:

The intersection of all intervals is the set of X that are in every interval. For each interval, X must be in [L_i, R_i] (if L_i <= R_i) or in [L_i, S] or [1, R_i] (if L_i > R_i).

So the intersection must be a range that is covered by all intervals.

To compute this, we can track the maximum of all L_i and the minimum of all R_i for the non-wrapping intervals. But for the wrapping intervals, perhaps we need to split them into two parts.

Alternatively, perhaps we can normalize the intervals.

Wait, here's a possible method:

For each interval [a, b], if a <= b, then it's a non-wrapping interval. Otherwise, it's a wrapping interval (from a to S and then from 1 to b).

The intersection of all intervals must be a range that is covered by all intervals, considering both wrapping and non-wrapping cases.

But this seems complicated. Let's think of the problem in terms of possible candidates for X.

But with S up to 2e9 (since A and B can be up to 1e9 each), we can't check each X individually.

So, perhaps the correct approach is to compute the minimal upper bound and maximal lower bound, but considering the circular nature.

Wait, here's a possible approach inspired by the fact that the intersection of all intervals must be non-empty.

Let’s compute for all D_i the interval [L_i, R_i], where L_i = (-D_i + 1) mod S, R_i = (L_i + A - 1) mod S.

Then, for the intersection to be non-empty, there must be a point X that is in all [L_i, R_i] intervals (considering the circular wrapping).

To find this, we can compute the maximum of all L_i and the minimum of all R_i. If the maximum L_i <= the minimum R_i, then the interval [max_L, min_R] is the intersection. But if this interval is empty (max_L > min_R), then there is no intersection.

But this is only valid if all intervals are non-wrapping. If some intervals are wrapping, this approach may not work.

Wait, but perhaps the intervals can be transformed into a linear space by "unrolling" the circle. For example, imagine the circle is split into a line from 1 to S, and then repeated. So each interval can be represented as a range on this line. The intersection would be the set of points that are covered by all intervals in at least one of their repetitions.

But this seems complicated.

Alternatively, perhaps the problem can be transformed into a maximum and minimum problem.

Let me think again. For each interval [L_i, R_i], which is either a non-wrapping interval (L_i <= R_i) or a wrapping interval (L_i > R_i). For the intersection to be non-empty, there must be a point X that is in all intervals.

But how can we model this?

Another approach is to note that the intersection of all intervals is the set of X that are allowed by all D_i. For each D_i, X must be in [L_i, R_i] (if L_i <= R_i) or in [L_i, S] or [1, R_i] (if L_i > R_i).

The key is to find the maximal lower bound and minimal upper bound for X.

But perhaps the correct way is to compute the maximum of all L_i and the minimum of all R_i. If the maximum L_i <= the minimum R_i, then the interval [max_L, min_R] is the intersection. Otherwise, there is no intersection.

But this approach doesn't account for wrapping intervals. For example, suppose some intervals are wrapping. Let's see:

Suppose S = 7, A = 2, B =5. So S=7.

Sample Input 1: 3 2 5, D = [1,2,9]

For D=1: L_i = (-1 +1) mod 7 = 0 mod 7 → 0, but since X is 1-based, perhaps the calculation is ( -D_i +1) mod S. Wait, let's compute L_i for D_i=1:

L_i = (-1 +1) mod 7 = 0 mod 7 → 0. But since X is in 1..7, perhaps the mod is adjusted to 7. So 0 mod 7 is 0 → which would correspond to 7?

Wait, maybe the correct formula for L_i is ( -D_i +1 ) mod S. But since X is 1-based, perhaps when the mod is 0, it's equivalent to S.

So for D_i=1:

L_i = ( -1 +1 ) mod 7 = 0 → which is equivalent to 7.

R_i = L_i + A -1 = 7 + 2-1 = 8 → 8 mod 7 = 1.

So the interval is [7, 1], which is a wrapping interval. So allowed X's are 7 and 1.

Wait, but according to the sample input 1, when X is 7 (the 7th day, which is a weekday), then D_i=1 would be 7+1=8 → 8 mod 7 is 1, which is a holiday. So the allowed X for D_i=1 is X=7, because (7 +1) mod 7 = 1, which is a holiday. So the interval for D_i=1 is [7, 1], which includes X=7, 1.

But according to the formula, L_i is 7, R_i is 1. So the interval is from 7 to 1, which wraps around. So the allowed X's are 7, 1.

Wait, but in the sample input 1, the X is 7, which is allowed for D_i=1.

So for the interval [7, 1], the allowed X's are 7, 1.

So in this case, the interval is a wrapping interval. So the allowed X's are X >=7 or X <=1.

But according to the formula, the allowed X's for D_i=1 are 7 and 1. But how does that fit into the interval [7, 1]?

Ah, perhaps the interval [L_i, R_i] is considered as a circular interval. So when L_i <= R_i, it's the normal interval. When L_i > R_i, it's the union of [L_i, S] and [1, R_i].

So for each D_i, the allowed X's are in the interval [L_i, R_i] if L_i <= R_i, else in [L_i, S] ∪ [1, R_i].

So the problem reduces to finding if the intersection of all such intervals is non-empty.

But how to compute the intersection of N such intervals efficiently?

An alternative approach is to model the allowed X's as the interval [max_L, min_R], where max_L is the maximum of all L_i, and min_R is the minimum of all R_i. But this only works if all intervals are non-wrapping. If some intervals are wrapping, this approach may not work.

Wait, but perhaps the correct approach is to compute two possibilities:

1. The intersection is a non-wrapping interval. So the maximum L_i <= the minimum R_i. Then, any X in [max_L, min_R] is allowed.

2. The intersection is a wrapping interval. So the X must be in [max_L, S] and [1, min_R]. For this to be possible, all intervals must include this wrapping interval.

But how to check this?

Alternatively, the maximum L_i and minimum R_i can be computed. If the maximum L_i <= the minimum R_i, then the intersection is [max_L, min_R], which is non-empty. Otherwise, the intersection is the union of [max_L, S] and [1, min_R], but only if this union is covered by all intervals.

Wait, perhaps this is the key idea. Let me explain:

Let max_L be the maximum of all L_i, min_R be the minimum of all R_i.

If max_L <= min_R, then the intersection is [max_L, min_R], which is non-empty.

Otherwise, the intersection is the set of X that are in [max_L, S] and in [1, min_R], but this requires that for every interval, either:

- The interval is a wrapping interval (L_i > R_i), and [max_L, S] is covered by [L_i, S] and [1, R_i] is covered by [1, R_i], or

- The interval is a non-wrapping interval, but [max_L, S] and [1, min_R] are subsets of [L_i, R_i].

But this seems complicated.

Alternatively, when max_L > min_R, the possible X's are those in [max_L, S] and [1, min_R]. For this to be a valid intersection, all intervals must cover both [max_L, S] and [1, min_R]. So for each interval [L_i, R_i], it must cover [max_L, S] (if L_i <= R_i, then L_i <= max_L and R_i >= S, which is impossible since R_i <= S. So in this case, if max_L > min_R, then all intervals must be wrapping intervals, and their L_i <= max_L and R_i >= min_R.

Wait, perhaps when max_L > min_R, then the intersection is [max_L, S] ∩ [1, min_R], which is the union of the two ranges. But for this to be non-empty, the ranges must overlap, which they do only if max_L <= S and min_R >=1, which is always true. But the problem is whether all intervals cover this union.

So for the case when max_L > min_R, the intersection is the union [max_L, S] and [1, min_R]. For this to be a valid intersection, every interval must cover this entire union.

An interval [L_i, R_i] (which is a wrapping interval, since L_i > R_i) covers [L_i, S] and [1, R_i]. So for the union [max_L, S] and [1, min_R] to be covered by all intervals, each interval must have L_i <= max_L and R_i >= min_R.

Because:

For the interval [L_i, R_i] (which is a wrapping interval), it covers [L_i, S] and [1, R_i]. For this to include [max_L, S] and [1, min_R], we need L_i <= max_L (so that [max_L, S] is within [L_i, S]), and R_i >= min_R (so that [1, min_R] is within [1, R_i]).

So in this case, the conditions are:

max_L <= min_R (case 1) → intersection is [max_L, min_R]

OR

max_L > min_R and for all intervals, L_i <= max_L and R_i >= min_R (case 2) → intersection is [max_L, S] ∪ [1, min_R]

If either case holds, then the answer is Yes.

So the algorithm is:

Compute max_L = maximum of all L_i

Compute min_R = minimum of all R_i

If max_L <= min_R, then check if there exists at least one X in [max_L, min_R]. Since the interval is non-empty (max_L <= min_R), this is possible. So answer is Yes.

Else, check if for all intervals, L_i <= max_L and R_i >= min_R. If this is true, then the union [max_L, S] and [1, min_R] is covered by all intervals, so answer is Yes.

Otherwise, answer is No.

So the steps are:

1. For each D_i, compute L_i = (-D_i +1) mod S. If L_i is 0, set it to S.

   Similarly, R_i = (L_i + A -1) mod S. If R_i is 0, set to S.

   Wait, but how to compute L_i and R_i correctly?

Wait, let's compute L_i as ( -D_i +1 ) mod S. But since we want it in the range 1..S, if the mod is 0, we need to set it to S.

For example, if S=7, D_i=1:

- D_i +1 = 0 → mod 7 is 0 → L_i = 7.

Similarly, R_i = (7 + 2-1) mod 7 → (7+1) mod7 = 1 → R_i=1.

So for D_i=1, L_i=7, R_i=1.

So the code for L_i and R_i should be:

L_i = (-D_i +1) % S

if L_i ==0: L_i = S

R_i = (L_i + A -1) % S

if R_i ==0: R_i = S

Wait, but (-D_i +1) can be negative. For example, D_i=2:

- D_i +1 = -1 → mod S (7) is 6. So L_i=6.

So the code would correctly compute L_i as (-D_i +1) mod S, then adjust to 1-based.

So the code for L_i and R_i would be:

S = A + B

for each D in D_list:

    L = (-D + 1) % S

    if L ==0:

        L = S

    R = (L + A -1) % S

    if R ==0:

        R = S

    # now, L and R are in 1..S

So for each D_i, L and R are computed as above.

Then, compute max_L = max of all L_i

min_R = min of all R_i

If max_L <= min_R: answer is Yes

Else:

   check if for all i, L_i <= max_L and R_i >= min_R

   if yes: answer is Yes

   else: No

So this is the approach.

Let's test this with sample input 1.

Sample Input 1:

3 2 5 → S=7

D = [1,2,9]

Compute for each D_i:

D=1:

L = (-1 +1) mod7 =0 → 7

R = (7 +2-1) mod7 = (8) mod7=1 →1

→ L=7, R=1

D=2:

L = (-2+1) mod7 = (-1) mod7=6 →6

R=6+2-1=7 mod7=0 →7

→ L=6, R=7

D=9:

L = (-9 +1) mod7 = (-8) mod7= (-8 +7*2) =6 →6 mod7=6

R=6+2-1=7 mod7=0 →7

So L=6, R=7

So for the three D_i:

max_L = max(7,6,6) =7

min_R = min(1,7,7) =1

So 7 <=1 → false.

So we need to check if for all i, L_i <= max_L (7) and R_i >= min_R (1).

Check each D_i:

For D=1: L=7 <=7, R=1 >=1 → yes.

For D=2: L=6 <=7, R=7 >=1 → yes.

For D=9: L=6 <=7, R=7 >=1 → yes.

So all intervals satisfy L_i <= max_L and R_i >= min_R. So answer is Yes.

Which matches sample input 1.

Sample Input 2:

2 5 10 → S=15

D=10,15.

Compute for each D_i:

D=10:

L = (-10 +1) mod15 = (-9) mod15=6 →6

R=6 +5-1=10 →10 mod15=10 →10.

So L=6, R=10.

D=15:

L = (-15+1) mod15= (-14) mod15=1 →1

R=1 +5-1=5 →5.

So L=1, R=5.

max_L = max(6,1)=6

min_R= min(10,5)=5.

6 <=5 → no.

Check if for all i, L_i <=6 and R_i >=5.

For D=10: L=6 <=6, R=10 >=5 → yes.

For D=15: L=1 <=6, R=5 >=5 → yes.

So the answer would be Yes, but the sample output is No. Wait, this contradicts.

Wait sample input 2's output is No, but according to this approach, it would be Yes.

Wait, what's wrong here?

Wait sample input 2:

Sample Input 2:

2 5 10

10 15

Sample Output 2: No

Let me compute according to the algorithm.

S=15, A=5.

For D=10:

L = (-10+1) mod15 = (-9) mod15=6 →6

R=6+5-1=10 →10 mod15=10.

So interval is [6,10].

For D=15:

L = (-15+1) mod15= (-14 mod15) →1.

R=1+5-1=5 →5.

So interval is [1,5].

max_L = max(6,1) =6.

min_R = min(10,5) =5.

6 >5, so check if all intervals have L_i <=6 and R_i >=5.

For D=10's interval [6,10], L=6 <=6, R=10 >=5 → yes.

For D=15's interval [1,5], L=1 <=6, R=5 >=5 → yes.

So according to the algorithm, answer is Yes. But sample output is No.

But this contradicts. So there's a mistake in the algorithm.

Hmm. So what's wrong here?

Wait, let's see the sample input 2:

The plans are on days 10 and 15.

We need to find if there's a starting day X such that:

X +10 days is a holiday, and X +15 days is a holiday.

Holidays are days 1-5, weekdays 6-15.

So S=15.

For X to be a day such that:

(X +10) mod15 is in 1-5.

and (X +15) mod15 is in 1-5.

But (X +15) mod15 = X mod15. So the second condition is X mod15 in 1-5.

The first condition is (X +10) mod15 in 1-5.

So X mod15 must be in 1-5 (from second condition).

So substituting into the first condition:

X mod15 +10 mod15 → (X +10) mod15.

If X is in 1-5:

X=1 → 1+10=11 mod15=11 → not in 1-5.

X=2 →12 → no.

X=3 →13 → no.

X=4 →14 → no.

X=5 →15 → mod15=0 → 15 → no.

So no X in 1-5 satisfies (X+10) mod15 in 1-5.

Hence, the answer is No.

But according to the algorithm, the answer would be Yes, which is incorrect.

So the algorithm is wrong.

So what's the problem with the algorithm?

The algorithm says that when max_L > min_R, we check if all intervals have L_i <= max_L and R_i >= min_R. If yes, then the answer is Yes.

But in sample input 2, the intervals are [6,10] and [1,5].

max_L is 6, min_R is5.

So for each interval:

First interval [6,10] → L_i=6 <=6, R_i=10 >=5 → yes.

Second interval [1,5] → L_i=1 <=6, R_i=5 >=5 → yes.

So according to the algorithm, answer is Yes. But actual answer is No.

So the algorithm is incorrect. So why?

Because the intersection of the two intervals is [6,10] and [1,5], but their intersection is empty. So the algorithm's logic is flawed.

Ah, I see. The algorithm's step 2 is incorrect. The step where it checks if all intervals have L_i <= max_L and R_i >= min_R is not sufficient to ensure that the union [max_L, S] and [1, min_R] is covered by all intervals.

In sample input 2, the intervals are [6,10] and [1,5]. The max_L is 6, min_R is5.

The algorithm's step 2 checks if all intervals have L_i <=6 and R_i >=5.

For the first interval [6,10], L_i=6 <=6, R_i=10 >=5 → yes.

For the second interval [1,5], L_i=1 <=6, R_i=5 >=5 → yes.

But the intersection of the two intervals is empty. Because [6,10] and [1,5] do not overlap.

But according to the algorithm, the answer is Yes, but the correct answer is No.

So the algorithm's logic is incorrect. Therefore, the approach needs to be revised.

So what's wrong with the algorithm's logic?

The algorithm assumes that if all intervals cover [max_L, S] and [1, min_R], then the union of those two ranges is the intersection. But this is not the case.

In sample input 2, the two intervals are [6,10] and [1,5]. The max_L is 6, min_R is5. The algorithm's step 2 says that the answer is Yes because all intervals cover [max_L, S] (which is 6-15) and [1,5], but the first interval [6,10] does not cover [1,5]. Wait, no. The first interval [6,10] is a non-wrapping interval. So it covers 6-10. The second interval [1,5] is also a non-wrapping interval. So the intersection of the two intervals is empty. So the algorithm's step 2 is incorrect.

So the algorithm's step 2 is not sufficient. The problem is that the algorithm's step 2 allows for intervals that do not actually cover the union [max_L, S] and [1, min_R].

So what's the correct way to model this?

Alternative approach:

The correct way to compute the intersection is:

If max_L <= min_R, then the intersection is [max_L, min_R], which is non-empty.

Else, the intersection is the set of X such that X is in [max_L, S] and X is in [1, min_R] for all intervals.

But how can this be possible?

Wait, in the case where max_L > min_R, the intersection would be the set of X that are in all intervals. Each interval can be either non-wrapping or wrapping.

For X to be in the intersection, X must be in every interval. So for each interval, X must be in [L_i, R_i] if it's non-wrapping, or in [L_i, S] ∪ [1, R_i] if it's wrapping.

So when max_L > min_R, the possible X's are those in [max_L, S] and [1, min_R]. But for this to be the case, every interval must include both [max_L, S] and [1, min_R].

But how to check this?

For each interval, it must cover [max_L, S] and [1, min_R].

So for an interval [L_i, R_i], which is non-wrapping (L_i <= R_i), it must cover [max_L, S] and [1, min_R]. But a non-wrapping interval can't cover [max_L, S] unless L_i <= max_L and R_i >= S. But R_i <= S (since it's non-wrapping), so R_i >= S implies R_i = S. So for a non-wrapping interval to cover [max_L, S], it must be that L_i <= max_L and R_i = S.

Similarly, for the interval to cover [1, min_R], it must be that L_i <= min_R and R_i >= min_R. But since the interval is non-wrapping, this would require L_i <= min_R <= R_i. But min_R is the minimum of all R_i. So if there's a non-wrapping interval with R_i < min_R, then it's impossible.

This seems complicated.

Alternative idea: when max_L > min_R, then the only possible X's that can be in the intersection are those in [max_L, S] and [1, min_R]. But for this to be possible, every interval must include all X in [max_L, S] and all X in [1, min_R].

So for each interval, [max_L, S] must be a subset of the interval, and [1, min_R] must be a subset of the interval.

So for each interval, the interval must cover [max_L, S] and [1, min_R].

So how to check this for each interval?

For an interval [L_i, R_i], which is non-wrapping (L_i <= R_i):

- To cover [max_L, S], we must have L_i <= max_L and R_i >= S. But R_i <= S (since it's non-wrapping), so R_i must be S. So L_i <= max_L and R_i = S.

- To cover [1, min_R], we must have L_i <=1 and R_i >= min_R. But since the interval is non-wrapping, this requires L_i <=1 and R_i >= min_R.

But this is only possible if min_R <= R_i, which is true since min_R is the minimum of all R_i.

But this seems very restrictive.

Alternatively, for the interval [L_i, R_i] (non-wrapping), to cover [max_L, S] and [1, min_R], the interval must be [1, S]. Which is only possible if the interval is the entire week. But that's not the case.

This approach seems to be leading to a dead end.

Alternative plan:

Perhaps the initial approach is incorrect and we need to think differently.

Each D_i imposes that X must be in a certain interval. The intersection of all these intervals must be non-empty.

Each interval is a set of X's that satisfy (X + D_i) mod S <= A.

But since X is modulo S, the problem is to find if the intersection of all the intervals is non-empty.

But how to compute this intersection.

Another approach is to model the allowed X's for each D_i as a range. For example, for each D_i, the allowed X's are:

(X + D_i) mod S <= A.

Which can be rewritten as X mod S <= (A - D_i) mod S.

Wait, let's see:

(X + D_i) mod S <= A.

But (X + D_i) mod S can be written as (X mod S + D_i mod S) mod S.

But this may not help.

Alternatively, let's consider that (X + D_i) mod S is in [1, A].

Which is equivalent to:

1 <= (X + D_i) mod S <= A.

Let’s denote S = A + B.

Let’s compute for each D_i the possible X mod S.

So for each D_i, X mod S must be in [ (1 - D_i) mod S, (A - D_i) mod S ].

Wait, but this is a circular interval.

For example, if D_i=1, then X mod S must be in [0, A-1] mod S. But since X is 1-based, 0 corresponds to S.

So for D_i=1, X must be in [S, A-1] if A-1 >= S, which is not possible. So perhaps the interval is [ (1 - D_i) mod S, (A - D_i) mod S ].

But this is a circular interval. So for each D_i, the allowed X mod S is in the interval [ (1 - D_i) mod S, (A - D_i) mod S ].

But since the interval is circular, this can be represented as a range that may wrap around.

So for each D_i, the allowed X's are those that are in the interval [L, R], where L = (1 - D_i) mod S, R = (A - D_i) mod S.

But this interval is [L, R] if L <= R, or [L, S-1] and [0, R] if L > R.

But since X is 1-based, we need to adjust this.

Wait, perhaps it's easier to model X as 0-based. Let's try that.

Let’s represent X as 0-based (0 to S-1). Then, the holiday days are 0 to A-1 (since the first day is 1 in 1-based, which is 0 in 0-based).

Then, for each D_i, the condition is (X + D_i) mod S <= A-1.

So (X + D_i) mod S <= A-1.

Let’s denote X as 0-based.

Then, for each D_i, the allowed X's are those X such that X <= (A-1 - D_i) mod S.

Wait, let's rearrange the inequality:

X + D_i <= A-1 + k*S for some integer k.

But since X is in 0..S-1, D_i is up to 1e9, but we can take modulo S.

So (X + D_i) mod S <= A-1.

Which is equivalent to X mod S <= (A-1 - D_i) mod S.

Wait, no. Let me think:

Let’s compute (X + D_i) mod S = (X mod S + D_i mod S) mod S.

Let’s denote D_i mod S as d.

So (X + d) mod S <= A-1.

This can be rewritten as X <= (A-1 - d) mod S.

But this is not straightforward.

Alternatively, the inequality (X + d) mod S <= A-1 is equivalent to:

X + d <= A-1 + k*S for some integer k.

But since X is in 0..S-1 and d is in 0..S-1, X +d can be up to 2S-2.

So possible k can be 0 or 1.

So:

If X +d <= A-1 → X <= A-1 -d.

Or X +d >= S and X +d <= S + A-1 → X >= S -d and X <= S + A-1 -d.

But since X <= S-1, the second condition becomes X >= S -d and X <= S + A-1 -d.

But S + A-1 -d = (A-1) + (S -d).

But S = A + B, so S -d = A + B -d.

Not sure.

Alternatively, the inequality (X +d) mod S <= A-1 can be written as:

(X +d) mod S is in [0, A-1].

Which is equivalent to X +d ∈ [k*S, k*S + A-1] for some integer k.

But since X is in [0, S-1], and d is in [0, S-1], X +d can be in [0, 2S-2].

So possible k can be 0 or 1.

So:

For k=0:

X +d <= A-1 → X <= A-1 -d.

But X must be >=0.

So allowed X's are 0 <= X <= min(A-1 -d, S-1).

But since A-1 -d can be negative, this interval may be empty.

For k=1:

X +d >= S → X >= S -d.

And X +d <= S + A-1 → X <= S + A-1 -d.

But X <= S-1, so the upper bound is min(S + A-1 -d, S-1).

So allowed X's are max(S -d, 0) <= X <= min(S + A-1 -d, S-1).

But S + A-1 -d = (A-1) + (S -d) = (A-1) + (A + B -d).

Not sure.

But combining the two cases:

The allowed X's are:

X <= A-1 -d or X >= S -d and X <= S + A-1 -d.

But since X is in 0..S-1, the second interval is X >= S -d and X <= S-1 (since S + A-1 -d >= S-1 only if A-1 -d >= -1, which is always true since A >=1 and d >=0).

So the allowed X's are:

[0, A-1 -d] ∪ [S -d, S-1]

But this is only if A-1 -d >=0, otherwise the first interval is empty.

So for each D_i, the allowed X's (0-based) are:

if A-1 -d >=0:

   [0, A-1 -d] ∪ [S -d, S-1]

else:

   [S -d, S-1]

But d = D_i mod S.

So for each D_i, we compute d = D_i % S.

Then, the allowed X's are:

if (A-1 -d) >=0:

   X is in [0, A-1 -d] or [S -d, S-1]

else:

   X is in [S -d, S-1]

But this seems complicated to handle for all D_i.

But the key is to find the intersection of all such intervals.

But with N up to 2e5, we need an O(N) approach.

Alternative idea: For each D_i, the allowed X's are a union of two intervals. The intersection must be non-empty.

But how to compute the intersection of all such intervals.

Alternative approach: Let's think in terms of the allowed X's for each D_i.

For each D_i, the allowed X's are the set of X that satisfy (X + D_i) mod S <= A-1 (0-based). So:

Let’s denote d_i = D_i % S.

Then, (X + d_i) mod S <= A-1.

This can be rewritten as X <= (A-1 - d_i) mod S.

Wait, no. Let's think again.

The condition (X + d_i) mod S <= A-1.

This is equivalent to X + d_i <= A-1 + k*S for some integer k.

But since X is between 0 and S-1, and d_i is between 0 and S-1, X +d_i can be between 0 and 2S-2.

So k can be 0 or 1.

For k=0:

X +d_i <= A-1 → X <= A-1 -d_i.

But since X >=0, this interval is [0, min(A-1 -d_i, S-1)].

But if A-1 -d_i <0, this interval is empty.

For k=1:

X +d_i >= S → X >= S -d_i.

And X +d_i <= S + A-1 → X <= S + A-1 -d_i.

But X <= S-1, so the interval is [S -d_i, min(S + A-1 -d_i, S-1)].

But S + A-1 -d_i = (A-1) + (S -d_i) → since S = A + B, this is (A-1) + (A + B -d_i) = 2A + B -1 -d_i. But this may be larger than S-1.

So the interval is [S -d_i, S-1].

So combining both cases, the allowed X's for D_i are:

If A-1 -d_i >=0:

   [0, A-1 -d_i] ∪ [S -d_i, S-1]

Else:

   [S -d_i, S-1]

So for each D_i, the allowed X's are either one or two intervals.

The intersection of all such intervals must be non-empty.

But how to compute this intersection efficiently.

This seems very challenging, but perhaps we can find the minimal and maximal possible X in each interval and compute the intersection.

Alternatively, perhaps the correct approach is to find the maximum of all the lower bounds of the intervals and the minimum of all the upper bounds.

But considering that each interval can be a union of two ranges, this is not straightforward.

Alternative Idea:

Let’s consider that for the intersection to be non-empty, there must be at least one X that is allowed by all D_i.

For X to be allowed by all D_i, for each D_i, X must be in the allowed intervals of D_i.

So for each D_i, X must be in either the first interval [0, a_i] or the second interval [b_i, S-1], where a_i = A-1 -d_i and b_i = S -d_i.

But a_i can be negative, in which case the first interval is empty.

So for each D_i, the allowed X's are:

if a_i >=0: X <= a_i or X >= b_i

else: X >= b_i

So the problem reduces to finding an X that satisfies for all D_i:

X <= a_i or X >= b_i (if a_i >=0)

OR

X >= b_i (if a_i <0)

So how can we find such an X?

This seems like a problem that can be modeled as a set of constraints that must be satisfied by X.

Let’s consider all D_i where a_i >=0. For these, X must be <= a_i or >= b_i.

For D_i where a_i <0, X must be >= b_i.

So the constraints are:

For all D_i:

if a_i >=0:

   X <= a_i or X >= b_i

else:

   X >= b_i

We need to find an X in 0..S-1 that satisfies all constraints.

So how to find such an X?

This seems like a problem that can be modeled as follows:

The X must be >= the maximum of all (b_i where a_i <0) and also >= the maximum of all (b_i where a_i >=0 and X >= b_i is chosen) and <= the minimum of all (a_i where a_i >=0 and X <= a_i is chosen).

But this is not straightforward.

Alternative Idea:

Let’s find the maximum of all lower bounds and the minimum of all upper bounds.

The lower bounds are:

For D_i where a_i >=0: the lower bound is b_i (if we choose the X >= b_i option) or 0 (if we choose X <= a_i option).

For D_i where a_i <0: the lower bound is b_i.

The upper bounds are:

For D_i where a_i >=0: the upper bound is a_i (if we choose X <= a_i option) or S-1 (if we choose X >= b_i option).

For D_i where a_i <0: upper bound is S-1.

But this is too vague.

Alternative Idea:

Let’s consider that for each D_i where a_i >=0, the constraint is X <= a_i OR X >= b_i. For these, we can model this as X must be in (-inf, a_i] OR [b_i, +inf).

For D_i where a_i <0, the constraint is X >= b_i.

The intersection of all these constraints must be non-empty.

So the possible X's must satisfy:

X >= max_b, where max_b is the maximum of all b_i for D_i where a_i <0, and for D_i where a_i >=0, either X <= a_i or X >= b_i.

But how to model this.

Let’s compute:

max_b = maximum of all b_i for D_i where a_i <0.

For D_i where a_i >=0, let's compute the minimum a_i and the maximum b_i.

Let’s denote:

min_a = minimum of all a_i where a_i >=0.

max_b2 = maximum of all b_i where a_i >=0.

So for X to satisfy all constraints:

X >= max_b.

And for each D_i where a_i >=0:

Either X <= a_i or X >= b_i.

So for these D_i, since X >= max_b, which may be >= b_i, we need to ensure that for all D_i where a_i >=0, X >= b_i or X <= a_i.

But since X >= max_b, which is >= all b_i for D_i where a_i <0, but for D_i where a_i >=0, their b_i may be less than max_b.

So for D_i where a_i >=0, if b_i <= max_b, then X >= max_b implies X >= b_i, so the constraint for D_i is automatically satisfied.

But if b_i > max_b, then X >= max_b but < b_i would require that X <= a_i.

But X >= max_b and <= a_i is possible only if max_b <= a_i.

So for D_i where a_i >=0 and b_i > max_b:

We must have X <= a_i.

But X >= max_b, so the constraint becomes max_b <= X <= a_i.

If a_i < max_b, then this interval is empty, so no solution.

So the algorithm can be:

1. Compute max_b = maximum of all b_i for D_i where a_i <0.

   If there are no such D_i, set max_b = 0.

2. For all D_i where a_i >=0:

   a. Compute b_i and a_i.

   b. If b_i <= max_b: this D_i's constraint is automatically satisfied, since X >= max_b >= b_i.

   c. Else: X must be <= a_i. So the constraint is X <= a_i.

3. So the new constraints are:

   X >= max_b.

   X <= min_a, where min_a is the minimum of all a_i from D_i where a_i >=0 and b_i > max_b.

4. If there are any D_i where a_i >=0 and b_i > max_b and a_i < max_b: then the interval [max_b, a_i] is empty, so no solution.

5. So compute the minimal a_i among those D_i where a_i >=0 and b_i > max_b. Let this be min_a.

   The X must be in [max_b, min_a].

   If max_b > min_a: no solution.

   Else: yes.

So the steps are:

- Compute max_b as the maximum of all b_i for D_i where a_i <0.

- For all D_i where a_i >=0:

   if b_i > max_b:

      collect a_i into a list.

- Compute min_a = minimum of this list.

- if there are any such D_i:

   if max_b > min_a: no solution.

   else: yes.

- else:

   if max_b <= S-1: yes.

But wait, what if there are no D_i where a_i >=0 and b_i > max_b?

Then, the constraints are X >= max_b, and for all other D_i (those with a_i >=0 and b_i <= max_b), their constraints are satisfied.

So X can be any value >= max_b, up to S-1.

So if max_b <= S-1: yes.

Else: no.

But max_b can be up to S-1.

For example, if max_b is S-1, then X can be S-1.

So in this case, yes.

So putting it all together:

The algorithm is:

Compute for each D_i:

d_i = D_i % S

a_i = A-1 - d_i

b_i = S - d_i

if a_i >=0:

   then this D_i has two intervals: [0, a_i] and [b_i, S-1]

else:

   this D_i has interval [b_i, S-1]

Compute max_b as the maximum of all b_i where a_i <0.

Compute the list of a_i where a_i >=0 and b_i > max_b.

Let this list be called 'a_list'.

If 'a_list' is not empty:

   min_a = min(a_list)

   if max_b > min_a: answer is No

   else: answer is Yes

Else:

   if max_b <= S-1: answer is Yes

   else: answer is No

Let's test this algorithm on sample input 2.

Sample Input 2:

N=2, A=5, B=10 → S=15.

D_i are 10 and 15.

For D=10:

d_i =10 %15=10

a_i =5-1 -10 =4-10= -6 → a_i <0.

So this D_i's interval is [b_i, 14], where b_i =15-10=5.

So for this D_i, a_i <0, so contribute to max_b.

max_b is 5.

For D=15:

d_i =15%15=0

a_i =5-1 -0=4 ≥0.

b_i =15-0=15 → but S=15, so 15 mod S is 0. So b_i=15 mod S=0.

Wait, wait, in the 0-based model, S=15, so X can be 0..14.

So b_i = S -d_i =15 -0=15 → mod S is 0.

So for D=15:

a_i =4 ≥0.

b_i =0.

So this D_i has a_i >=0.

Now, for this D_i, we check if b_i > max_b.

max_b is 5.

b_i=0 >5? No. So this D_i's b_i is not > max_b.

So this D_i's constraint is X <= a_i (4) or X >= b_i (0).

But since X >= max_b (5), the constraint X >=0 is automatically satisfied.

But X >=5 must be <=4 or >=0. But X >=5 and <=4 is impossible. So this D_i's constraint is not satisfied.

So according to the algorithm:

For D=15, a_i=4 >=0, b_i=0.

max_b is 5.

b_i=0 is not > max_b (5), so this D_i is not added to 'a_list'.

So 'a_list' is empty.

So the algorithm proceeds to check if max_b <= S-1.

max_b=5 <=14 → yes. So answer is Yes.

But sample input 2's correct answer is No.

So the algorithm is still incorrect.

Ah, here's the problem. The algorithm's logic is flawed because even though the 'a_list' is empty, there are D_i with a_i >=0 and b_i <= max_b, but their constraints may not be satisfied.

In sample input 2, the D=15's constraint is X <=4 or X >=0.

But X >=5 (max_b=5) must satisfy this constraint. X >=5 and (X <=4 or X >=0). Since X >=5, X <=4 is false, but X >=0 is true. So the constraint is satisfied.

But according to the algorithm, the answer is Yes, but the correct answer is No.

So why is this discrepancy?

Because the algorithm's logic is missing something.

Wait, let's re-examine the sample input 2.

In sample input 2:

For D=10: the allowed X's are [5,14] (since a_i <0).

For D=15: allowed X's are [0,4] or [0,14] (since a_i=4, b_i=0).

But the X must be in both allowed sets.

So the intersection is [5,14] (from D=10) intersected with ([0,4] or [0,14]).

The intersection is [5,14] ∩ [0,14] → [5,14].

But for D=15, the allowed X's are [0,4] or [0,14]. So X in [5,14] is allowed for D=15.

But according to the problem statement, the answer should be No.

Wait, why?

Wait, let's compute for X=5:

For D=10: X=5 is in [5,14] → allowed.

For D=15: X=5 must be in [0,4] or [0,14]. X=5 is in [0,14] → allowed.

So according to this, X=5 is allowed.

But according to the sample input 2's explanation, the answer is No.

Wait, sample input 2's explanation says:

Plans are on days 10 and 15.

Assuming X is 5 (0-based, which is 6 in 1-based):

For D=10: X+10=15 → 15 mod15=0 → 0-based is 0, which is a holiday (0-based holidays are 0-4).

So 0 is allowed.

For D=15: X+15=5+15=20 → 20 mod15=5 → 5 is a weekday (since A=5 in 1-based is 5 days, 0-based 0-4. 5 is in 0-based 5, which is a weekday).

So the plan on D=15 would be on a weekday, which is not allowed.

Ah, so the mistake is in the 0-based vs 1-based modeling.

So the error comes from the initial model.

The problem statement uses 1-based days. So in the 0-based model, the holidays are 0..A-1 (1-based 1..A), and weekdays are A..A+B-1 (1-based A+1..A+B).

But in the algorithm's 0-based model, the condition is (X + D_i) mod S <= A-1.

But in sample input 2:

For D=15, which is 15 days later.

In 1-based, X is the current day (1-based). Let's say X is 6 (0-based 5).

D=15: 6 +15=21 → 21 mod15=6 → 6 is a weekday (since A=5, holidays are 1-5, weekdays 6-15).

So the plan is on a weekday, which is not allowed.

But according to the 0-based model, X=5 (0-based) is 5, which is a holiday (0-based 0-4 are holidays). So X=5 (0-based) is a holiday. Wait, no. 0-based holidays are 0-4. X=5 is a weekday.

Ah, here's the mistake. The 0-based model's holiday is 0..A-1. So for A=5, the holidays are 0,1,2,3,4 (0-based), which are 1-5 in 1-based.

So X=5 (0-based) is a weekday.

So in the 0-based model, the algorithm's condition for D=15 is (X +15) mod15 <=4.

X=5: 5+15=20 mod15=5 →5 <=4? No. So the condition is not satisfied.

So X=5 is not allowed for D=15.

So why did the algorithm think it's allowed?

Because the algorithm's logic is incorrect.

The algorithm's logic for D=15:

d_i =15 mod15=0.

a_i =4 >=0.

b_i=15-0=15 mod15=0.

So the allowed X's for D=15 are [0,4] or [0,14].

Wait, no. In the 0-based model, for a_i=4 >=0, the allowed X's are [0,4] or [15-0,14] → [0,4] or [15,14], which is [0,4] or [0,14] (since 15 mod15=0).

So the allowed X's are [0,4] or [0,14], which is just [0,14]. So any X in 0-14 is allowed for D=15.

But this is not correct.

Ah, here's the mistake in the algorithm's model.

The algorithm's model for allowed X's when a_i >=0 is [0, a_i] ∪ [b_i, S-1].

But when a_i=4 and b_i=0 (S=15):

[0,4] ∪ [0,14] = [0,14]. So all X are allowed for D=15.

But according to the problem's condition, when X=5 (0-based), which is a weekday, D=15's plan is scheduled on a weekday.

So the algorithm's model is incorrect.

So the error is in the way the allowed X's are computed.

The condition (X +d_i) mod S <= A-1.

For D=15, d_i=0.

So (X +0) mod15 <=4.

So X mod15 <=4.

So X can be 0-4, 15-4=11? No.

Wait, X is 0-based, so X can be 0-14.

So (X +0) mod15 is X.

So the condition is X <=4.

So allowed X's are 0-4.

So the algorithm's model is incorrect.

The model says that for a_i=4 and b_i=0, the allowed X's are [0,4] ∪ [0,14], which is incorrect.

The correct allowed X's are [0,4].

So the mistake is in the model's calculation of the allowed X's.

So the algorithm's model for allowed X's when a_i >=0 is incorrect.

The correct allowed X's for a_i >=0 are:

Either X <= a_i OR X >= b_i.

But when a_i=4 and b_i=0:

X >=0 is always true. So the allowed X's are X <=4 OR X >=0 → which is all X.

But this is not correct.

So the model's logic is wrong.

So the error is in the way the allowed X's are computed.

The correct allowed X's for a_i >=0 is:

X <= a_i OR X >= b_i.

But when b_i is 0, X >=0 is always true, so the allowed X's are all X.

But this is not the case.

The condition is (X +d_i) mod S <= A-1.

For d_i=0, the condition is X mod S <= A-1 → X <= A-1.

So allowed X's are 0 <= X <= A-1.

So the model's logic is incorrect.

So the mistake is in the model's representation of the allowed X's.

The correct allowed X's for a_i >=0 is:

X <= a_i OR X >= b_i.

But in reality, when a_i >=0, the allowed X's are:

X <= a_i OR X >= b_i.

But when b_i is 0, this is X <= a_i OR X >=0 → which is all X <= a_i or X >=0, which is all X.

But this is not correct.

So the model's logic is incorrect.

The correct allowed X's for a_i >=0 is:

X <= a_i OR X >= b_i.

But when b_i is 0, this is X <= a_i OR X >=0 → which is X >=0 (since a_i >=0).

So this is all X >=0, which is all possible X.

But this is not correct.

So the model's logic is incorrect.

So the initial approach is wrong.

The correct way to compute the allowed X's for each D_i is:

For D_i, compute d_i = D_i mod S.

The condition is (X +d_i) mod S <= A-1.

This is equivalent to X mod S <= (A-1 -d_i) mod S.

But this is not the same as X <= a_i OR X >= b_i.

So the model's approach is incorrect.

So we need to find a new way to model the allowed X's.

Alternative Idea:

Let’s compute the allowed X's for each D_i as follows:

The condition is (X +d_i) mod S <= A-1.

Let’s denote this as:

X +d_i ≡ k mod S, where k is in [0, A-1].

So X ≡ (k -d_i) mod S.

So for each k in [0, A-1], X can be (k -d_i) mod S.

So the allowed X's are the set { (k -d_i) mod S | k=0,1,...,A-1 }.

But this set is a contiguous interval.

For example, if d_i=0, then allowed X's are 0,1,...,A-1.

If d_i=1, allowed X's are S-1, 0, 1, ..., A-2.

So the allowed X's form a contiguous interval.

So for each D_i, the allowed X's are a contiguous interval [L, R], where:

L = (0 -d_i) mod S

R = (A-1 -d_i) mod S

If L <= R: the interval is [L, R]

Else: the interval is [L, S-1] and [0, R]

So the allowed X's for each D_i are a single contiguous interval that may wrap around.

So the problem reduces to finding the intersection of N contiguous intervals on a circle (mod S).

The intersection is non-empty if there exists a point X that is in all intervals.

To compute this, we can use the following approach:

- For each interval, compute its start and end (L_i, R_i).

- The intersection of all intervals is the set of X that are in every interval.

- To compute this, we can track the maximum of all L_i and the minimum of all R_i.

- If the maximum L_i <= the minimum R_i, then the intersection is [max_L, min_R], which is non-empty.

- Else, the intersection is the set of X that are in [max_L, S-1] and in [0, min_R], but only if all intervals cover both ranges.

But how to check this.

This is similar to the initial approach but with correct interval representation.

So the algorithm is:

For each D_i:

   d_i = D_i % S

   L_i = (0 - d_i) % S

   R_i = (A-1 - d_i) % S

   if L_i <= R_i:

      the interval is [L_i, R_i]

   else:

      the interval is [L_i, S-1] and [0, R_i]

Compute max_L = maximum of all L_i

Compute min_R = minimum of all R_i

If max_L <= min_R:

   the intersection is [max_L, min_R], which is non-empty → answer is Yes

Else:

   check if all intervals include both [max_L, S-1] and [0, min_R]

   if yes → answer is Yes

   else → answer is No

But how to check if all intervals include both [max_L, S-1] and [0, min_R].

For an interval [L_i, R_i] (non-wrapping), it includes [max_L, S-1] and [0, min_R] only if L_i <= max_L and R_i >= min_R.

But for a non-wrapping interval, this requires L_i <= max_L and R_i >= min_R.

But if the interval is non-wrapping and max_L > R_i or min_R < L_i, then it does not cover both ranges.

For a wrapping interval (L_i > R_i), it covers [L_i, S-1] and [0, R_i]. So to cover [max_L, S-1] and [0, min_R], we need:

L_i <= max_L (to cover [max_L, S-1] within [L_i, S-1])

and R_i >= min_R (to cover [0, min_R] within [0, R_i])

So for all intervals, whether wrapping or not, the condition is:

L_i <= max_L and R_i >= min_R.

So the algorithm is:

Compute max_L and min_R.

If max_L <= min_R → Yes.

Else, check if for all intervals, L_i <= max_L and R_i >= min_R.

If yes → Yes.

Else → No.

So this is the same as the initial approach, but with the correct computation of L_i and R_i.

Let's test this on sample input 2.

Sample input 2:

N=2, A=5, B=10 → S=15.

D=10 and 15.

For D=10:

d_i =10 %15=10.

L_i = (0 -10) mod15 =5.

R_i = (4