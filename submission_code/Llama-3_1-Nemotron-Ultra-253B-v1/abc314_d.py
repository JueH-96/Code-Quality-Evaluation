<think>
Okay, let's see. I need to solve this programming problem. Let's read the problem carefully.

The problem is about performing a series of operations on a string. The operations can be of three types: changing a character, converting all uppercase to lowercase, or converting all lowercase to uppercase. After all operations, we need to output the final string.

Hmm. The constraints are tight. The string can be up to 5e5 characters, and there are up to 5e5 operations. So any solution that is O(Q*N) time is definitely going to be too slow. For example, if each type 2 or 3 operation requires iterating through the entire string each time, that's 5e5 * 5e5 operations, which is 2.5e11—way too slow.

So, the key is to find a way to handle the type 2 and 3 operations efficiently, without actually modifying the entire string each time. Because those operations are all about flipping the case of certain characters, maybe we can track the current state with some flags and apply the transformations lazily when needed.

Wait, here's an idea: instead of modifying the string immediately for type 2 and 3 operations, we can keep track of a global "shift" state. For example, if the current shift is 0, then the characters are as stored. If there's a type 2 operation (convert uppercase to lowercase), maybe we can represent that as a shift that affects how characters are read or written. But how?

Alternatively, maybe we can track two parameters: a lower_to_upper flag and an upper_to_lower flag. But combining multiple operations might get complicated. For instance, if you have a type 2 followed by a type 3, what's the net effect?

Wait, let's think about the operations:

- Type 2: convert all uppercase letters to lowercase. So after this operation, any uppercase letters become lowercase, and lowercase letters remain as they are.

- Type 3: convert all lowercase letters to uppercase. So after this, lowercase letters become uppercase, uppercase remain.

But applying these operations multiple times can be represented as a transformation that's applied on the fly. For example, if we have a current state where we have to invert the case of certain letters when we output them.

Wait, perhaps we can model the current case transformation as a combination of these operations. Let's say we have a variable that represents the current transformation. For example, each character in the stored string is in a certain case, but when we output, we apply the current transformation.

But how to handle the transformations when they are applied in sequence. Let's think:

Each type 2 operation is equivalent to: for each character in the string, if it's uppercase, make it lowercase. But if we have multiple type 2 operations, applying them again would not change anything. Similarly, type 3 operations convert all lowercase to uppercase, but applying it again does nothing.

But if we have a sequence of type 2 and type 3 operations, the order matters. For example:

Original string: 'A'

Type 3: convert lowercase to uppercase. But 'A' is already uppercase, so no change. Then type 2: convert to lowercase. Now it's 'a'.

Another example: if after a type 2, we have a type 3. Let's see:

Original: 'a'

Type 2: no change (since it's lowercase). Then type 3: convert to 'A'.

So the order of operations matters. But how can we model the current state of the transformations without applying them to the entire string each time?

Hmm. Let's think of the transformations as a function that, when we need to read or modify a character, we apply the current transformations to get the actual character.

Wait, maybe we can track two flags: one for whether we need to flip uppercase to lowercase (type 2), and another for flipping lowercase to uppercase (type 3). But when these flags are set, how do they interact?

Alternatively, let's model the current transformation as a function that, given a character, returns what it should be after applying all the pending transformations.

Wait, perhaps the key is to track the current state of the transformations. Let's see:

Each time a type 2 or type 3 operation is applied, instead of modifying the string, we can track a "shift" value. For example:

- Initially, the shift is 0: characters are as stored.

- When a type 2 operation is applied (convert uppercase to lowercase), the shift becomes 1. But wait, how does this affect subsequent operations?

Wait, maybe the shift can be represented as a combination of the operations. Let's think:

Suppose that after a series of type 2 and type 3 operations, the current transformation can be represented as a function that, when applied to a character, gives the desired case. For example:

If the current transformation is T, then for any character c in the stored string, the actual character is T(c).

So, for each type 2 operation, T becomes T' where T'(c) is c.lower() if c was uppercase, else c.

But if we have multiple type 2 operations, applying T once is sufficient. So, perhaps the current transformation can be represented as a state that is a combination of the operations applied so far.

Alternatively, the transformation can be represented as a sequence of operations, but that's not efficient. So we need a way to represent the current transformation as a function that can be applied on the fly when needed.

Wait, here's a better idea. Let's track two boolean variables:

- to_lower: indicates that any uppercase characters should be converted to lowercase.

- to_upper: indicates that any lowercase characters should be converted to uppercase.

But these two can't be both true at the same time, because if you have to_lower and to_upper both true, then for a character, which transformation takes precedence?

Wait, no. Let's think: suppose to_lower is true. Then, when we process a character, first we check if it's uppercase, and convert to lowercase. Then, if to_upper is also true, we check if it's lowercase and convert to uppercase. But that would result in the character being converted to uppercase again. But that's not the same as applying the operations in sequence.

Wait, the order in which the transformations are applied matters. For example, if we first apply type 2 (convert uppercase to lowercase), then type 3 (convert lowercase to uppercase), then the net effect is that all uppercase letters become lowercase and then to uppercase again (so no change?), but lowercase letters become uppercase.

Wait, let's take an example. Suppose original character is 'A' (uppercase). Apply type 2: becomes 'a'. Then apply type 3: 'a' becomes 'A'. So net effect is no change. But if the original character is 'a', type 2 does nothing, then type 3 converts to 'A'.

So, the combined effect of type 2 followed by type 3 is: all lowercase letters become uppercase, and uppercase letters are converted to lowercase then back to uppercase. So, the net effect is that all characters become uppercase. So, the combined effect is equivalent to a type 3 operation.

But how can we model this with flags?

Alternatively, perhaps the current transformation can be represented as a function that is the composition of the individual operations. But how to represent that efficiently.

Another approach: when a type 2 or type 3 operation is encountered, instead of modifying the entire string, we can track the current state of the transformations. Then, when a character is accessed (either for reading or modifying), we apply the current transformations to get the actual character.

But for that, we need to know the current state of the transformations. Let's model the current state with two flags:

- flip_upper_to_lower: if true, any uppercase character is converted to lowercase.

- flip_lower_to_upper: if true, any lowercase character is converted to uppercase.

But these two flags can be in any combination. However, applying these transformations in a certain order may lead to different results.

Wait, but perhaps the order of applying these transformations is important. For example, if we have both flags set, then for a character, we first check if it's uppercase and flip it to lowercase, then check if it's lowercase and flip to uppercase. So the order in which the transformations are applied would matter.

Alternatively, maybe the flags can be combined into a single transformation state. For example:

Each time a type 2 or type 3 operation is applied, we can compute the new transformation state.

Let me think of the transformation state as a function that takes a character and returns the transformed character. Then, when applying multiple operations, the functions are composed.

For example:

- Initial state: identity function.

- After type 2: the function is "convert uppercase to lowercase".

- After type 3: the function is "convert lowercase to uppercase".

If we apply type 2 followed by type 3, the composed function would be: for a character c, first apply type 2's function (convert to lowercase if uppercase), then apply type 3's function (convert to uppercase if lowercase). So:

If c is uppercase: converted to lowercase, then to uppercase → original c.

If c is lowercase: converted to lowercase (no change), then to uppercase → uppercase.

So the composed function is equivalent to converting all lowercase to uppercase. Which is the same as type 3's function. So in this case, the flags can be optimized.

Wait, but how to model this in code. Let's see.

Let me think of the current transformation as a combination of two steps: first, apply any uppercase to lowercase conversion, then apply any lowercase to uppercase conversion. Or maybe the order is the other way around.

Wait, the order in which the transformations are applied is crucial. Let's say we have two flags:

- to_lower: whether to convert uppercase to lowercase.

- to_upper: whether to convert lowercase to uppercase.

But the order of applying these two steps matters. For example, if to_lower is applied first, then to_upper.

So for a character c:

if to_lower is true, convert c to lowercase.

then, if to_upper is true, convert the result to uppercase.

So, the overall transformation would be:

c → lowercase(c) → uppercase(lowercase(c)).

But this is equivalent to:

if c was uppercase: lowercase(c) is lowercase, then uppercase → uppercase(c) → original c.

if c was lowercase: lowercase(c) is c, then uppercase → uppercase(c).

So the net effect is that all lowercase characters are converted to uppercase, and uppercase characters remain as uppercase. So this is equivalent to a type 3 operation.

But how to track this in code. Let's model the current transformation as a combination of these two steps. Then, whenever a new type 2 or type 3 operation is applied, we can update the transformation state accordingly.

For example:

Current state: (t_lower, t_upper), where t_lower is the number of times we need to apply the to_lower transformation, and t_upper is the number of times for to_upper. But since applying the same transformation twice cancels it out (e.g., converting to lowercase twice is the same as once), we can track t_lower as a boolean (whether it's been applied an odd number of times), and similarly for t_upper.

Wait, but the order in which they are applied matters. So perhaps the state is a sequence of transformations. But tracking the sequence would be expensive. So, perhaps the state can be represented by two booleans: whether the to_lower transformation is active, and whether the to_upper transformation is active. But the order in which they are applied must be considered.

Alternatively, the state can be represented as a function that is the composition of the transformations. But how to represent that efficiently.

Alternatively, here's a different approach. Let's think that each character in the original string is stored as-is, but when we need to access or modify it, we apply the current transformation state to get the actual character. Then, when a type 1 operation is performed (changing a character), we need to reverse the transformation to store the new character in the original form.

Wait, this might be a way to handle it. Let's see:

Suppose the original string is stored as a list, but each character is kept in its original form, and the current transformation state is tracked. When we need to read a character (e.g., for a type 1 operation, or when building the final output), we apply the current transformation to the stored character to get the actual value. However, when modifying a character (type 1), we need to reverse the transformation to store the new character in the original form.

Wait, maybe not. Let's think through an example.

Suppose the current transformation is to convert all uppercase to lowercase (type 2). So, the stored string has 'A', but when read, it becomes 'a'. Now, if a type 1 operation changes the 1st character to 'B', then the stored character should be 'B', but when the transformation is applied, it becomes 'b'. But the user intended to set the character to 'B' in the current state. Wait, no. The problem statement says that for a type 1 operation, the x_i-th character is changed to c_i. So, the operation is applied to the current state of the string. So, if the current transformation is active, the type 1 operation is modifying the transformed string.

Wait, this is a critical point. Let me re-read the problem statement.

For the type 1 operation: change the x_i-th character of S to c_i. So, the S here is the current S after all previous operations. So, when you perform a type 1 operation, you are modifying the current S, which includes any transformations from previous type 2 or 3 operations.

So, for example:

Original S is 'A'. Then, a type 2 operation is applied, so S becomes 'a'. Then, a type 1 operation changes the first character to 'B'. So, the S after this is 'B'. Then, if another type 3 operation is applied, S becomes 'b'.

Wait, but according to the problem statement, the type 1 operation is modifying the current S. So, after the type 2 operation, S is 'a'. Then type 1 changes it to 'B', so S is 'B'. Then type 3 converts all lowercase to uppercase. But 'B' is uppercase, so no change. So the final S is 'B'.

But if the type 1 operation is applied to the transformed S, then the stored data must reflect the current state of the string. So, in this case, after the type 2 operation, the stored S is 'a' (lowercase). Then, the type 1 operation changes it to 'B' (uppercase). So, the stored S is 'B'. Then, the type 3 operation converts all lowercase to uppercase, but 'B' is already uppercase, so no change.

So, the stored S must always reflect the current state after all operations. Therefore, type 2 and 3 operations must modify the stored string. But this would be O(N) per type 2 or 3 operation, which is not feasible for large N and Q.

So, this approach would not work for the given constraints. So, the problem is to find a way to represent the current state of the string without actually performing O(N) operations for each type 2 or 3.

So, back to the initial idea: track the current transformation state and apply it on the fly when needed.

Let's model the current transformation as a function that can be applied to a character. Then, for each type 2 or 3 operation, we update this function. For type 1 operations, we need to apply the inverse of the current transformation to the new character to store it in the original form. Then, when building the final string, we apply the current transformation to each character.

Wait, this might work. Let's think:

- We have a list of characters stored in their "original" form.

- We have a transformation function T that is applied to the original characters to get the current state.

- For a type 1 operation, we want to set the x-th character in the current state to c. So, to store this in the original form, we need to apply the inverse transformation T^{-1} to c. Then, when T is applied to the stored original character, it becomes c.

But how to invert the transformation function T.

Let's define T as a function that takes an original character and returns the current character. So, for example:

If T is the identity function (no transformations), then the original character is the same as the current.

If T is type 2 (convert uppercase to lowercase), then T(c) is c.lower() if c is uppercase, else c.

But the inverse would be: when we want to set the current character to c, we need to find the original character o such that T(o) = c.

But this is not straightforward, because T is not invertible. For example, if T converts uppercase to lowercase, then T('A') = 'a', and T('a') = 'a'. So, if the current character is 'a', the original could be 'A' or 'a'. So, the inverse is ambiguous.

This suggests that this approach may not work. Because when we perform a type 1 operation, we need to modify the original character such that when the transformation is applied, it becomes the desired c. But if the transformation is lossy (like converting to lowercase), we can't uniquely determine what the original character should be.

So, this approach might not be feasible.

Alternative idea: instead of storing the original characters, we can track the current state with some variables that represent the pending transformations, and apply them when necessary.

For example, let's track two variables:

- flip_upper: the number of times we need to flip uppercase letters to lowercase. Since applying this twice is the same as not applying it, we can track it as a boolean (whether the count is odd).

- flip_lower: the number of times we need to flip lowercase letters to uppercase. Again, a boolean.

But the order of applying these flips matters. For example, flipping uppercase to lowercase first, then flipping lowercase to uppercase would have a different effect than the reverse.

Wait, let's model the current state as a sequence of transformations. But since the order matters, and the number of possible states is limited (since applying the same transformation twice cancels it out), perhaps we can track the state as a combination of these two booleans and the order in which they are applied.

But how?

Alternatively, the current state can be represented by a function that is the composition of the transformations. For example, each type 2 operation adds a transformation to lowercase uppercase letters, and each type 3 adds a transformation to uppercase lowercase letters. But the order of these transformations is important.

But composing these functions can be simplified. Let's see:

Each transformation can be represented as a function that takes a character and returns the transformed character.

Let's define:

- T1: convert uppercase to lowercase. So, T1(c) = c.lower() if c is uppercase, else c.

- T2: convert lowercase to uppercase. So, T2(c) = c.upper() if c is lowercase, else c.

Then, applying T1 followed by T2 is equivalent to:

For a character c:

if c is uppercase: T1(c) is lowercase, then T2(c) is uppercase. So overall, c → uppercase.

if c is lowercase: T1(c) is lowercase, then T2(c) is uppercase. So overall, c → uppercase.

So, the composition of T1 followed by T2 is equivalent to converting all characters to uppercase.

Similarly, applying T2 followed by T1:

For uppercase c: T2(c) is uppercase (no change), then T1 converts to lowercase. So overall, uppercase → lowercase.

For lowercase c: T2 converts to uppercase, then T1 converts to lowercase. So overall, lowercase → lowercase.

So, the composition is equivalent to converting all uppercase to lowercase.

So, the order of transformations affects the outcome.

But how to track the current state of transformations as a combination of T1 and T2, in a way that allows us to compute the effective transformation for each character.

Hmm. Let's think of the current transformation as a list of the transformations applied, but since applying T1 twice is equivalent to not applying it, and same for T2, we can track the count of T1 and T2 modulo 2. But the order in which they are applied is still important.

Alternatively, the current transformation can be represented as a pair (a, b), where a is the number of T1 transformations applied modulo 2, and b is the number of T2 transformations applied modulo 2. But the order in which they are applied would require that the effective transformation is T1 applied a times, followed by T2 applied b times. But since T1 and T2 are not commutative, this would not capture all possible combinations.

Wait, but if the order of application is always T1 first, then T2, then the pair (a, b) can represent the current state. But this may not be the case.

Alternatively, perhaps the current state can be represented as the effective function that is the composition of all the transformations so far, but in a way that allows us to compute the function for each character quickly.

Let me think about the possible states. Each transformation is either T1 or T2. Let's see what the possible combinations are:

- No transformations: identity.

- T1: convert uppercase to lowercase.

- T2: convert lowercase to uppercase.

- T1 followed by T2: convert all to uppercase.

- T2 followed by T1: convert uppercase to lowercase, and lowercase to lowercase (i.e., same as T1).

Wait, no. Let's see:

Case 1: T1 followed by T2:

For a character c:

If c is uppercase → T1 makes it lowercase → T2 makes it uppercase. So overall, c remains uppercase.

If c is lowercase → T1 leaves it → T2 makes it uppercase. So overall, c becomes uppercase.

So the composition is equivalent to T2.

Wait, that's interesting. So T1 followed by T2 is equivalent to T2.

Wait, no. Because for uppercase characters, the result is the same as original. For lowercase, it's converted to uppercase. So the composition is equivalent to converting all lowercase to uppercase, which is exactly T2. So, applying T1 then T2 is the same as just applying T2.

Wait, that's a key insight. So, if you apply T1 and then T2, the net effect is T2.

Similarly, applying T2 and then T1: let's see.

For uppercase c: T2 leaves it as uppercase → T1 converts to lowercase. So overall, uppercase becomes lowercase.

For lowercase c: T2 converts to uppercase → T1 converts to lowercase. So overall, lowercase becomes lowercase.

So the composition is equivalent to T1.

So, applying T2 followed by T1 is equivalent to T1.

So, this suggests that the order of applying T1 and T2 can be simplified. For example:

- Applying T1 followed by T2 is equivalent to T2.

- Applying T2 followed by T1 is equivalent to T1.

So, the order of applying T1 and T2 can be optimized.

This is a crucial insight. So, the current state can be represented by the last transformation applied. Because applying T1 and T2 in a certain order may result in the same as the last transformation.

Wait, let's see:

If the current state is T1, and then we apply T2, the new state is T2.

If the current state is T2, and then we apply T1, the new state is T1.

If the current state is T1, and we apply T1 again, the state becomes identity.

Similarly for T2.

So, the possible states are:

- Identity (no transformation).

- T1 (convert uppercase to lowercase).

- T2 (convert lowercase to uppercase).

Because applying T1 and T2 in sequence results in the last transformation.

So, the state can be represented as one of these three possibilities.

Wait, let's test this:

Case 1: Apply T1, then T1 again. The net effect is identity.

Case 2: Apply T2, then T2 again. Net effect is identity.

Case 3: Apply T1, then T2. Net effect is T2.

Case 4: Apply T2, then T1. Net effect is T1.

Case 5: Apply T1, then T2, then T1. Net effect is T1.

Case 6: Apply T2, then T1, then T2. Net effect is T2.

So, the state can be represented as the last applied transformation (T1, T2, or none).

Wait, but what about applying T1, then T2, then T1 again? The first two steps give T2, then applying T1 gives T1.

So, the state transitions are:

- Current state is identity:

   - Apply T1 → state is T1.

   - Apply T2 → state is T2.

- Current state is T1:

   - Apply T1 → state is identity.

   - Apply T2 → state is T2.

- Current state is T2:

   - Apply T1 → state is T1.

   - Apply T2 → state is identity.

So, the state can be represented as one of three possible values: 0 (identity), 1 (T1), 2 (T2). Then, each type 2 or 3 operation changes the state according to these transitions.

This is a key insight. So, the current state can be tracked with a single variable (state) that can be 0, 1, or 2.

But wait, what about when the state is T1, and a type 3 operation (which is T2) is applied. Then, the new state is T2. Then, applying a type 2 operation (T1) again would transition to T1.

So, the state transitions are as follows:

When applying a type 2 operation (T1):

   state = state.apply(T1)

When applying a type 3 operation (T2):

   state = state.apply(T2)

The apply function for the state would be:

- If current state is identity:

   - applying T1 → state becomes T1.

   - applying T2 → state becomes T2.

- If current state is T1:

   - applying T1 → state becomes identity.

   - applying T2 → state becomes T2.

- If current state is T2:

   - applying T1 → state becomes T1.

   - applying T2 → state becomes identity.

So, the state can be represented with a single variable, and each type 2 or 3 operation toggles the state according to these rules.

This is a manageable approach. So, the current state is tracked as one of the three possible states. Then, when we need to read or modify a character, we apply the current state's transformation to get the actual character.

But how does this affect the type 1 operations?

For a type 1 operation, we are given a position x and a character c. The operation is to set the x-th character of the current string to c. But the current string is the original string with the current state's transformation applied.

So, to store the new character in the original string, we need to apply the inverse transformation of the current state to c. Because when the current state is applied to the original character, it becomes the current character. So, if we want the current character to be c, the original character should be the inverse transformation applied to c.

But what is the inverse of the current state's transformation?

Let's think:

The current state's transformation is a function T. To store a new character c in the original string, we need to find a character o such that T(o) = c. Then, o is the inverse transformation applied to c.

But the function T may not be invertible. For example, if T is T1 (convert uppercase to lowercase), then T(o) = c implies that o is either c (if c is lowercase) or o is the uppercase version of c (if c is lowercase). Wait, no. If T is T1, then:

T(o) = o.lower() if o is uppercase, else o.

So, if T is T1, and we want T(o) = c, then:

If c is lowercase:

   o can be either lowercase c or uppercase C.

If c is uppercase:

   o must be uppercase C, and T(o) would be lowercase c. But in this case, c is uppercase, so T(o) would be lowercase, which is not equal to c. So, it's impossible. So, in this case, if the current state is T1, and we want to set a character to uppercase C, it's impossible because T1 would convert it to lowercase c.

But according to the problem statement, the type 1 operation allows us to set the character to any case. So, how is this possible?

Wait, the problem statement says that for the type 1 operation, we change the x_i-th character of S to c_i. Here, S is the current string after all previous operations. So, the current state's transformation has already been applied to the original string to get S. So, when we perform a type 1 operation, we are directly modifying the current S. Therefore, the original string and the current state must be such that the current state's transformation is applied to the original string to get S.

But if we want to modify S directly, then the original string must be stored in a way that allows us to apply the current state's transformation to get S. However, this would require that for each type 2 or 3 operation, we have to apply the transformation to the entire string, which is O(N) time and thus not feasible.

So, this brings us back to the original problem: how to handle type 2 and 3 operations efficiently.

But with the state tracking approach, perhaps we can model the original string as being stored in a way that is modified by the state. For example:

The original string is stored as a list of characters. The current state is tracked as a variable. When a type 1 operation is performed, we need to compute what the original character should be so that when the current state is applied, it becomes the desired c.

But this requires that the transformation is invertible. Let's see:

For the current state, which is one of T0 (identity), T1 (convert uppercase to lowercase), T2 (convert lowercase to uppercase):

- If the current state is T0, then the original character is the same as the current character. So, to set the current character to c, we set the original character to c.

- If the current state is T1, then the current character is T1(original character). So, to set the current character to c, we need to set the original character to a value o such that T1(o) = c.

But T1(o) is o.lower() if o is uppercase, else o.

So, for T1(o) = c:

If c is lowercase:

   o can be any character that when converted to lowercase is c. So, o can be c (lowercase) or uppercase C.

But in this case, the original character can be either lowercase or uppercase. But how to choose which one to store?

Wait, the problem statement allows us to set the current character to c, regardless of the current state. So, the original character should be set to a value that, when the current state is applied, results in c.

But when the current state is T1, the transformation is: convert uppercase to lowercase, leave lowercase as is.

So, to get T1(o) = c, o must be:

If c is lowercase: o can be c (lowercase) or uppercase C. But T1(C) would be c.

But we need to choose o such that T1(o) = c. So, if c is lowercase, then o can be either lowercase c or uppercase C. But when the current state is T1, the original character is stored as o, and the current character is T1(o) = c.

But when the state changes, the original characters are transformed again. So, if we set o to uppercase C, then if the state changes to T0, the current character would be C. But according to the type 1 operation, the current character should be set to c (lowercase). So, in this case, we need to set o to lowercase c, not uppercase C.

Wait, no. Because when the current state is T1, the current character is T1(o). So, to have the current character be c, we must set o to a value such that T1(o) = c.

If c is lowercase, then o can be either lowercase c or uppercase C. But T1(C) is c. So, in this case, we can choose to store o as C. Then, when the state is T1, the current character is c. But if the state changes to T0, the current character would be C. But the type 1 operation is supposed to set the current character to c at that time. So, this suggests that the choice of o must be such that T1(o) = c, but when the state changes, the current character would change accordingly.

But according to the problem statement, the type 1 operation modifies the current character, not the original. So, the original must be stored in a way that when the current state is applied, the result is the desired c.

But how?

Let me think with an example:

Current state is T1. The current character is 'a' (because the original is 'A' or 'a').

Now, a type 1 operation changes it to 'B'. So, the current character must become 'B' (uppercase). But according to T1, the original character would be stored as 'B', because T1('B') is 'b', which is not the desired 'B'. So, this approach doesn't work.

Ah, here's the problem. If the current state is T1, which converts uppercase to lowercase, then the current character can't be uppercase. So, how can a type 1 operation set it to an uppercase character when the current state is T1?

This suggests that the initial approach of tracking the state and applying transformations on the fly is not sufficient. Because in this case, the type 1 operation is impossible. But according to the problem statement, it is allowed.

Wait, the problem statement says that in the type 1 operation, you can set the x_i-th character to any character (uppercase or lowercase). So, even if the current state is T1 (which converts uppercase to lowercase), the type 1 operation can set the character to uppercase. But according to the current state's transformation, that uppercase character would be converted to lowercase. So, the type 1 operation must modify the original character in such a way that after applying the current state's transformation, the result is the desired character.

But in this case, if the current state is T1, and the desired character is 'B' (uppercase), then the original character must be 'B' (uppercase), because T1('B') is 'b', which is not the desired 'B'. So, this is impossible.

Wait, this suggests that there's a contradiction. But according to the problem statement, it's allowed. So, there must be a misunderstanding here.

Wait, no. Let's re-examine the problem statement.

The problem statement says that in the type 1 operation, you change the x_i-th character of S to c_i. Here, S is the current string after all previous operations. So, the current state's transformation has already been applied. So, the type 1 operation is modifying the current S, not the original. Therefore, the current state's transformation must be applied to the original string to get S. Then, modifying S's character would require that the original string is modified in such a way that when the current state's transformation is applied, the result is the new S.

But this is only possible if the current state's transformation is invertible. But as we saw earlier, T1 is not invertible. For example, if the current state is T1, and S is 'a', then the original character could be 'A' or 'a'. So, when you perform a type 1 operation to change the character to 'B', you need to set the original character to 'B', but when T1 is applied, it becomes 'b', not 'B'. So, this is impossible.

This suggests that the initial approach is incorrect, and that we need to find a different way to model the problem.

Wait, but according to the sample input 1:

Sample Input 1:

7

AtCoder

5

1 4 i

3 0 a

1 5 b

2 0 a

1 4 Y

Sample Output 1: atcYber

Let's walk through the steps:

Initial S: AtCoder (original string is 'AtCoder').

First operation: type 1, change 4th character to 'i'. So, the 4th character is 'C' (index 3 if 0-based, but the problem uses 1-based). So, the original string becomes 'AtiCoder'? Or wait, the original string is modified directly.

Wait, the problem statement says that the operations are performed on the string S. So, each operation modifies S. So, the initial approach of applying each operation directly to S is correct, but it's O(N) per type 2 or 3 operation, which is not feasible for large N and Q.

But for the sample input, after the first operation, the string is 'AtCider' (the 4th character is changed to 'i'). Then, the second operation is type 3, which converts all lowercase to uppercase. So, the string becomes 'ATCIDER'. The third operation is type 1, changing the 5th character to 'b', so 'ATCIbER'. Then, type 2 converts all uppercase to lowercase: 'atciber'. Then, type 1 changes the 4th character to 'Y', resulting in 'atcYber'.

So, the problem is that the type 1 operations are applied to the current S, which is the result of all previous operations. So, the original approach of storing the current S and applying each operation directly would work, but it's O(N) per type 2 or 3 operation, which is not feasible for large N and Q.

But given the constraints (N and Q up to 5e5), we need a solution that handles all operations in O(1) or O(log N) time per operation.

So, the challenge is to find a data structure or a way to represent the string and the operations such that each operation can be handled in O(1) time, and the final string can be generated in O(N) time.

Let's think about the following approach:

We can track the current state of the string with two parameters: a flip_upper flag and a flip_lower flag, which indicate whether uppercase letters should be flipped to lowercase and vice versa. However, the order of flipping is important.

But how to model the current state such that the type 1 operations can be handled correctly.

Alternative idea: for each character in the original string, we can track its original case, and also track the current flip state. Then, when we need to read or modify a character, we can compute the current case based on the flip state.

But how to represent the flip state. Let's say we have two global flags:

- flip_upper: if true, uppercase letters are converted to lowercase.

- flip_lower: if true, lowercase letters are converted to uppercase.

But the order of flipping is important. For example, if both flags are true, then:

For an uppercase character, first flip_upper converts it to lowercase, then flip_lower converts it to uppercase again.

For a lowercase character, first flip_upper has no effect, then flip_lower converts it to uppercase.

So, the net effect is that all letters are converted to uppercase.

But how to track these flags and apply them in the correct order.

Alternatively, the flags can be applied in a specific order. For example, first apply flip_upper, then flip_lower.

So, the current state is determined by the two flags and the order in which they are applied.

But how to manage the flags when type 2 or 3 operations are applied.

Let's model the flags as follows:

- When a type 2 operation is applied (convert uppercase to lowercase), we toggle the flip_upper flag.

- When a type 3 operation is applied (convert lowercase to uppercase), we toggle the flip_lower flag.

But this is not correct, because applying type 2 twice would toggle the flag twice, which is equivalent to not applying it. So, the flags are boolean variables indicating whether the corresponding transformation is active.

But the order of applying the transformations is always flip_upper first, then flip_lower.

So, for any character, the current value is computed as:

current_char = original_char

if flip_upper and current_char is uppercase:
    current_char = current_char.lower()
if flip_lower and current_char is lowercase:
    current_char = current_char.upper()

So, the transformations are applied in the order of flip_upper first, then flip_lower.

Now, when a type 2 operation is applied, it toggles the flip_upper flag. When a type 3 operation is applied, it toggles the flip_lower flag.

But how does this model the actual transformations.

Let's test this model with some examples.

Example 1: Apply type 2 (toggle flip_upper to True).

Original character is 'A' (uppercase).

current_char = 'A'

flip_upper is True: convert to lowercase → 'a'.

flip_lower is False: no change.

Result: 'a'.

Example 2: Apply type 3 (toggle flip_lower to True).

Original character is 'a' (lowercase).

current_char = 'a'

flip_upper is False: no change.

flip_lower is True: convert to uppercase → 'A'.

Result: 'A'.

Example 3: Apply type 2 followed by type 3.

flip_upper is True, flip_lower is True.

Original character is 'A' → after flip_upper → 'a' → flip_lower → 'A'.

Original character is 'a' → flip_upper no change → 'a' → flip_lower → 'A'.

So, the net effect is that all characters are converted to uppercase.

Which is the same as applying type 3.

Another example: Apply type 3 followed by type 2.

flip_upper is True, flip_lower is True.

Original character is 'A' → flip_upper → 'a' → flip_lower → 'A'.

Original character is 'a' → flip_upper no change → 'a' → flip_lower → 'A'.

Same as before.

So, in this model, the order of applying type 2 and type 3 operations is irrelevant. The final flags are the same, and the net effect is the same.

But according to the problem statement, the order of operations does matter. For example:

Original string is 'a'.

Apply type 2 (convert uppercase to lowercase) → no change.

Apply type 3 → convert lowercase to uppercase → 'A'.

So, the final string is 'A'.

In the model, applying type 2 toggles flip_upper to True. Then applying type 3 toggles flip_lower to True.

So, for the original character 'a', the current_char is computed as:

flip_upper is True: 'a' is lowercase, so no change.

flip_lower is True: convert to uppercase → 'A'.

Which is correct.

Another example: Original string is 'A'.

Apply type 3 (toggle flip_lower to True). Then apply type 2 (toggle flip_upper to True).

For the original character 'A':

flip_upper is True: convert to 'a'.

flip_lower is True: 'a' is lowercase → convert to 'A'.

So, the final character is 'A'.

But according to the problem statement, the type 3 operation converts lowercase to uppercase (but the original character is uppercase, so no change). Then type 2 converts uppercase to lowercase. So, the final character is 'a'.

But according to the model, the final character is 'A'.

This is a discrepancy. So, the model is incorrect.

Ah, here's the problem. The model's order of applying flip_upper and flip_lower is not correct. In the problem statement, type 2 operations convert uppercase to lowercase, and type 3 convert lowercase to uppercase. The order in which these operations are applied affects the result.

In the example above:

Original character is 'A'.

Apply type 3 (convert lowercase to uppercase): since the character is uppercase, no change. So, the string remains 'A'.

Apply type 2 (convert uppercase to lowercase): the character becomes 'a'.

So, the correct result is 'a'.

But according to the model:

flip_upper is True (from type 2), flip_lower is True (from type 3).

Processing the original character 'A':

flip_upper converts to 'a'.

flip_lower converts to 'A'.

So, the model's result is 'A', which is incorrect.

So, the model's order of applying flip_upper first and then flip_lower is not correct. The order should be the reverse: apply flip_lower first, then flip_upper.

Wait, let's think again. Let's reverse the order of applying the transformations.

Model': for each character, apply flip_lower first, then flip_upper.

So, current_char = original_char

if flip_lower and current_char is lowercase:
    current_char = current_char.upper()
if flip_upper and current_char is uppercase:
    current_char = current_char.lower()

Now, let's test the previous example.

Original character 'A'.

Apply type 3: toggle flip_lower to True.

Apply type 2: toggle flip_upper to True.

Processing:

current_char = 'A'

flip_lower is True: current_char is uppercase → no change.

flip_upper is True: current_char is uppercase → convert to lowercase → 'a'.

So, the final result is 'a', which is correct.

Another example: apply type 2 followed by type 3.

Original character 'a'.

type 2: toggle flip_upper to True.

type 3: toggle flip_lower to True.

Processing:

current_char = 'a'

flip_lower is True: convert to uppercase → 'A'.

flip_upper is True: convert to lowercase → 'a'.

So, the final result is 'a'.

But according to the problem statement, type 2 converts uppercase to lowercase (original is 'a' → no change), then type 3 converts lowercase to uppercase → 'A'. So the correct result is 'A', but the model's result is 'a'.

So, this approach also has discrepancies.

This suggests that the order in which the transformations are applied is crucial and that the model must apply the transformations in the order of the operations.

But how to track the order of the transformations.

This brings us back to the initial idea of tracking the current state as a combination of the transformations in the order they were applied. But this seems complex.

Alternative approach: instead of tracking the transformations, track the current case of each character. But this would require O(N) time for type 2 or 3 operations.

But given the constraints, this is not feasible.

Alternative idea: since type 2 and type 3 operations affect all characters, and type 1 operations are point updates, perhaps we can use a segment tree with lazy propagation. But this would be complicated and may not be necessary.

Wait, let's think about the following:

Each type 2 or type 3 operation toggles a global state. The current state can be represented by two flags: whether we have to flip uppercase to lowercase (type 2) and whether we have to flip lowercase to uppercase (type 3). But the order of flipping must be considered.

But applying type 2 and type 3 operations in sequence can be represented as a combination of these flags and the order in which they are applied.

But this seems difficult.

Alternative approach inspired by the sample code:

We can track the current state as a combination of the transformations, and for each type 1 operation, we can compute the original character that would result in the desired current character after applying the current state's transformations.

But how.

Let's define the current state as a function T that takes a character and returns the transformed character.

For each type 2 operation, T is modified to T' = T followed by the type 2 transformation.

For each type 3 operation, T is modified to T' = T followed by the type 3 transformation.

Then, for a type 1 operation, to set the current character to c, we need to compute the original character o such that T(o) = c. Then, we store o in the original string.

But the problem is that T may not be invertible, so there may be multiple possible o's. But according to the problem statement, the type 1 operation must set the current character to c, so we need to find an o such that T(o) = c. If there are multiple possibilities, we can choose any, but when the state changes, the original character o will be transformed according to the new state.

But how to find o such that T(o) = c.

Let's model T as a function that can be composed, and find its inverse.

For example, T can be a function that is the composition of several type 2 and type 3 transformations. To find the inverse function, we need to reverse the order of the transformations and apply their inverses.

But this is getting complicated.

Alternatively, let's track the current state as a list of transformations, and for each type 1 operation, apply the inverse transformations in reverse order to c to get o.

But this would require storing the list of transformations, which could be up to Q in length, leading to O(Q) time per type 1 operation, which is not feasible.

But given that applying type 2 followed by type 3 is equivalent to type 3, and type 3 followed by type 2 is equivalent to type 2, the list of transformations can be reduced to a single transformation.

Wait, earlier analysis shows that applying type 2 followed by type 3 is equivalent to type 3. So, the current state can be represented by the last transformation applied.

So, the current state can be one of:

- identity (no transformation)

- type 2 (convert uppercase to lowercase)

- type 3 (convert lowercase to uppercase)

So, for example:

If the current state is type 2, and a type 3 operation is applied, the new state is type 3.

If the current state is type 3, and a type 2 operation is applied, the new state is type 2.

If the current state is type 2, and a type 2 operation is applied, the new state is identity.

So, the state can be represented with a single variable, which can be 0 (identity), 1 (type 2), or 2 (type 3).

Now, for a type 1 operation, given the current state, we need to compute the original character o such that applying the current state's transformation to o results in the desired character c.

Let's consider each possible state:

Case 1: state is identity.

Then, o must be c.

Case 2: state is type 2.

The transformation is: convert uppercase to lowercase.

So, T(o) = o.lower() if o is uppercase, else o.

We need to find o such that T(o) = c.

So:

If c is uppercase:

   Then, T(o) = c implies that o must be uppercase and o.lower() = c. But c is uppercase, so o.lower() is c.lower(), which is lowercase. So, this is impossible. Therefore, in this case, there is no solution. But according to the problem statement, the type 1 operation must allow setting the character to any c. So, this suggests that the model is incorrect.

But this is a contradiction. So, the initial approach must be wrong.

This indicates that the current state must be tracked in a way that allows type 1 operations to set the character to any c, regardless of the current state.

This brings us back to the original problem: the only way to handle type 1 operations correctly is to apply the current state's transformation to the original string to get the current string, modify the current string, and then store the new original string as the modified current string with the inverse transformation applied.

But this requires that the transformation is invertible.

But type 2 and type 3 transformations are not invertible. So, this approach is not feasible.

Alternative idea: instead of storing the original string, store the current string. Then, type 2 and type 3 operations can be applied directly to the current string. But this would require O(N) time per type 2 or 3 operation, which is not feasible for large N.

But given the constraints of up to 5e5 operations, this approach would result in O(Q*N) time, which is 2.5e11 operations—way too slow.

So, this approach is not feasible.

So, what's the solution?

The key insight is to realize that type 2 and type 3 operations are commutative in a certain way, and that their effect can be represented with a combination of two global flags, and the order of applying the flags can be managed.

Let's try to model the current state with two flags:

- flip_upper: indicates whether uppercase letters should be converted to lowercase.

- flip_lower: indicates whether lowercase letters should be converted to uppercase.

But the order of applying these flags is crucial. For example, if both flags are true, then:

For an uppercase letter:

   first, flip_upper converts it to lowercase.

   then, flip_lower converts it to uppercase.

For a lowercase letter:

   first, flip_upper has no effect.

   then, flip_lower converts it to uppercase.

So, the net effect is that all letters are uppercase.

But according to the problem statement, applying type 2 followed by type 3 would result in all letters being uppercase.

But according to the model, the flags would be flip_upper = True and flip_lower = True, and the order of application is flip_upper first, then flip_lower.

But in this case, the model correctly captures the net effect.

Now, let's model the type 2 and type 3 operations as follows:

- Type 2 operation: toggle flip_upper.

- Type 3 operation: toggle flip_lower.

Then, when applying a type 2 or type 3 operation, we simply toggle the corresponding flag.

Now, for a type 1 operation, we need to compute what the original character should be so that when the current flags are applied, the result is the desired c.

But how?

Let's think of the current character as a function of the original character and the flags:

current_char = original_char

if flip_upper:
    current_char = current_char.lower() if current_char.isupper() else current_char

if flip_lower:
    current_char = current_char.upper() if current_char.islower() else current_char

But we need to find the original_char such that after applying the current flags, the result is c.

So, given c, we need to find original_char such that:

original_char is modified by the current flags to become c.

Let's consider all possible cases based on the flags and the desired c.

Case 1: flip_upper is False, flip_lower is False.

Then, original_char must be c.

Case 2: flip_upper is True, flip_lower is False.

The transformation is: uppercase → lowercase, lowercase → unchanged.

So, to get current_char = c:

if c is lowercase:

   original_char can be uppercase (C) or lowercase (c). But when flip_upper is applied, C becomes c, and c remains c.

   So, to get c, original_char can be c or C.

   But when the flags change, the original_char will be transformed again. So, to ensure that the current_char is c, we need to set original_char to a value that when transformed with the current flags gives c.

   So, if flip_upper is True, and we want current_char to be c (lowercase), then original_char can be c (lowercase) or uppercase C.

   But if we set original_char to uppercase C, then when flip_upper is toggled (becomes False), the current_char would be C, which may not be desired.

   However, according to the problem statement, the type 1 operation is to set the current_char to c, regardless of future operations.

   So, the correct approach is to set original_char to a value that, when the current flags are applied, results in c.

   So, in this case, if flip_upper is True and c is lowercase:

   original_char can be c (lowercase) or uppercase C.

   But we need to choose one. To minimize future changes, perhaps we should store the original_char as c (lowercase), because that way, if flip_upper is toggled, the character remains c.

   Or, perhaps we should store it as uppercase C, so that when flip_upper is applied again, it becomes lowercase c.

   But this depends on the desired behavior.

   However, according to the problem statement, the type 1 operation sets the current character to c. The original_char must be set such that when the current flags are applied, the result is c.

   So, in this case, original_char can be either c or uppercase C. But we need to choose one.

   Let's choose the one that, when the current flags are applied, gives c.

   So, if flip_upper is True, and we want current_char to be c (lowercase), then:

   original_char can be c (lowercase), because flip_upper leaves it as c.

   Or original_char can be uppercase C, which flip_upper converts to c.

   So, either choice is valid. But to ensure that the original_char is stored in a way that is compatible with future operations, we need to choose the original_char that, when the current flags are applied, becomes c.

   So, in this case, we can set original_char to c.lower() if it's uppercase, but this may not be correct.

   Alternatively, we can compute the original_char by applying the inverse transformation.

   The inverse transformation of flip_upper is to toggle it again. So, to find the original_char, we can apply the inverse of the current flags.

   So, the inverse of the current flags is to toggle flip_upper and flip_lower.

   Wait, perhaps the way to compute the original_char is to apply the inverse transformation to c.

   The current transformation is: apply flip_upper, then flip_lower.

   The inverse transformation would be to apply the inverse of flip_lower, then the inverse of flip_upper.

   But what is the inverse of each flag?

   The inverse of flip_upper is to toggle it again. So, if the current flag is flip_upper = True, the inverse is to set it to False, and apply the inverse transformation.

   But this is getting complicated.

   Let's think differently. For each type 1 operation, given the current flags, we need to compute what the original_char should be so that when the flags are applied, it becomes c.

   So, we can model this as:

   original_char = c

   for each transformation in reverse order:

       apply the inverse transformation to original_char

   So, the transformations are applied in the order flip_upper, then flip_lower. To invert this, we need to apply the inverse of flip_lower, then the inverse of flip_upper.

   But what is the inverse of each transformation?

   The inverse of flip_upper (convert uppercase to lowercase) is to convert lowercase to uppercase, but only if the original character was uppercase.

   This is not straightforward.

   Alternatively, for each transformation, we can compute the inverse for a given c.

   Let's consider the current flags and compute the original_char.

   For example:

   Current flags: flip_upper = True, flip_lower = False.

   Desired current_char is c.

   The transformation is: uppercase → lowercase, lowercase → unchanged.

   So, to get c:

   if c is lowercase:

       original_char can be lowercase c or uppercase C.

   if c is uppercase:

       original_char must be uppercase C, and flip_upper must be False. But since flip_upper is True, this is impossible.

   So, in this case, if c is uppercase, it's impossible to have current_char be c when flip_upper is True.

   But according to the problem statement, it is allowed. So, this suggests that the model is incorrect.

   This indicates that the approach of using two flags and applying them in a certain order is not sufficient to handle all cases.

   Therefore, the correct approach must be to represent the current state as a combination of the transformations, and for type 1 operations, apply the inverse transformation to the desired c to get the original_char.

   But this requires that the transformation is invertible, which is only possible if the transformation is bijective.

   However, the transformations are not bijective. For example, the flip_upper transformation converts both 'A' and 'a' to 'a'.

   So, the inverse transformation is not unique.

   Therefore, the only way to handle type 1 operations correctly is to apply the transformation to the original string to get the current string, modify the current string, and then store the new original string as the modified current string with the inverse transformation applied.

   But this requires O(N) time for each type 2 or 3 operation, which is not feasible.

   So, given the time constraints, this suggests that the intended solution is to model the current state with two flags and apply the transformations on the fly when reading or writing characters.

   But how to handle the type 1 operations correctly.

   Let's try to proceed with the model where the current state is represented by two flags: flip_upper and flip_lower, and the order of application is flip_upper first, then flip_lower.

   For type 1 operations:

   Given the desired current_char c, we need to compute the original_char o such that applying the current flags to o results in c.

   Let's write a function to compute o based on c and the current flags.

   The function would be:

   def compute_original(c, flip_upper, flip_lower):

       # Apply the inverse transformations in reverse order.

       # The current transformations are flip_upper followed by flip_lower.

       # So, the inverse is to apply the inverse of flip_lower, then the inverse of flip_upper.

       # Inverse of flip_lower: if flip_lower is True, then during the transformation, lowercase is converted to uppercase.

       # So, during inversion, if flip_lower is True, then to get c, the original after flip_upper must have been a lowercase character that was converted to uppercase.

       # Or, the original after flip_upper was uppercase and not converted.

       # This is getting complicated.

       # Let's think step by step.

       # The transformation is:

       # o -> flip_upper -> flip_lower -> c

       # We need to find o such that after applying flip_upper and then flip_lower, we get c.

       # So, let's first reverse the flip_lower step.

       # After flip_upper is applied, we have a character m.

       # Then, applying flip_lower to m gives c.

       # So, m is the character after flip_upper, before flip_lower.

       # So, m = the character that, when flip_lower is applied, gives c.

       # Then, o is the character that, when flip_upper is applied, gives m.

       # So, first, find m from c and flip_lower.

       # Then, find o from m and flip_upper.

       # So:

       m = c

       if flip_lower:

           # flip_lower converts lowercase to uppercase.

           # So, during the transformation, m was converted to c.

           # So, if c is uppercase:

           # m could have been lowercase (converted to uppercase) or uppercase (unchanged).

           # But during the transformation, m was converted to uppercase only if it was lowercase.

           # So, to find m, given c is uppercase:

           # m could be lowercase (c.lower()) or uppercase (c).

           # But how to choose.

           # We need to choose m such that when flip_lower is applied, it becomes c.

           # So, if c is uppercase:

           # m can be any character that when flip_lower is applied gives c.

           # So, m can be c (uppercase) or c.lower() (lowercase).

           # But during the transformation, m is the result of applying flip_upper to o.

           # So, we need to choose m such that when flip_upper is applied to o, it gives m.

           # So, the problem is to find o such that:

           # 1. flip_upper applied to o gives m.

           # 2. flip_lower applied to m gives c.

           # So, first, find m from c and flip_lower.

           # If flip_lower is True:

           # m can be either c (if m was uppercase) or c.lower() (if m was lowercase).

           # But during the transformation, m is the result after flip_upper.

           # So, m can be anything, but when flip_lower is applied, it becomes c.

           # So, if flip_lower is True:

           # m after flip_upper must be such that applying flip_lower gives c.

           # So, if flip_lower is True:

           # if c is uppercase:

           # m can be c (uppercase) or c.lower() (lowercase).

           # But during the transformation, m is the result of flip_upper applied to o.

           # So, let's proceed.

           # We need to find m such that when flip_lower is applied to m, we get c.

           # So:

           if c.isupper():

               # m could be c (uppercase) or c.lower() (lowercase).

               # But during the transformation, m is the result of flip_upper applied to o.

               # So, m can be either uppercase or lowercase.

               # We need to choose m such that when flip_upper is applied to o, we get m.

               # So, for example, if m is uppercase, then o must be uppercase (since flip_upper would convert it to lowercase only if it was uppercase).

               # Wait, no. flip_upper converts uppercase to lowercase. So, if o is uppercase, m is lowercase. If o is lowercase, m is lowercase.

               # So, m is the result of applying flip_upper to o.

               # So, m can be:

               # if o is uppercase: m = lowercase o.

               # if o is lowercase: m = o.

               # So, m is always lowercase.

               # Wait, no. Because flip_upper is applied first. So, m is the result after applying flip_upper.

               # So, m is o.lower() if o is uppercase, else o.

               # So, m is always lowercase or the same as o if o is lowercase.

               # So, m can be lowercase or lowercase (if o is uppercase) or lowercase (if o is lowercase).

               # So, m is always lowercase.

               # Wait, no. If o is lowercase, then flip_upper leaves it as lowercase. So, m is lowercase.

               # So, m is always lowercase.

               # But then, applying flip_lower to m (which is lowercase) converts it to uppercase.

               # So, c must be uppercase.

               # So, in this case, m is lowercase, and applying flip_lower to m gives uppercase c.

               # So, m must be c.lower().

               # Because m is lowercase, and flip_lower converts it to uppercase.

               # So, m = c.lower().

               # Then, we need to find o such that applying flip_upper to o gives m = c.lower().

               # So, o can be:

               # if flip_upper is True:

               # applying flip_upper to o gives m = c.lower().

               # So, o is either uppercase (which when converted to lowercase is m) or lowercase (which is m).

               # So, o can be m (lowercase) or uppercase M (where M.lower() = m).

               # So, to find o, given m = c.lower(), and flip_upper is True:

               # o can be m (lowercase) or uppercase M.

               # But when flip_upper is applied to o, it must result in m.

               # So, if o is uppercase M, then flip_upper converts it to m.

               # If o is lowercase m, then flip_upper leaves it as m.

               # So, in either case, o can be either uppercase or lowercase.

               # So, to choose o, we can choose either.

               # But to minimize future changes, perhaps we should choose o as m (lowercase), so that if flip_upper is toggled, the character remains m.

               # So, in this case, o = m = c.lower().

               # So, the original_char o is c.lower().

               # But wait, let's test this.

               # Example: c is 'A' (uppercase), flip_upper is True, flip_lower is True.

               # We want to find o such that:

               # 1. apply flip_upper to o: if o is uppercase, it becomes lowercase; if lowercase, remains.

               # 2. apply flip_lower to the result: if lowercase, convert to uppercase.

               # So, the final result is 'A'.

               # So, the transformation steps:

               # o → flip_upper → m → flip_lower → 'A'.

               # So, m must be lowercase 'a'.

               # So, o can be 'A' (uppercase) → flip_upper → 'a' → flip_lower → 'A'.

               # Or o can be 'a' (lowercase) → flip_upper → 'a' → flip_lower → 'A'.

               # So, in both cases, the final result is 'A'.

               # So, for type 1 operation, to set the current_char to 'A', we can choose o to be 'A' or 'a'.

               # But according to the model, when we compute_original('A', True, True), we would:

               # m = 'A'.lower() = 'a'.

               # Then, compute o based on m and flip_upper.

               # Since flip_upper is True, o can be 'a' (lowercase) or 'A' (uppercase).

               # So, choose o = 'a' (lowercase).

               # So, when the current flags are applied, o is 'a' → flip_upper leaves it as 'a' → flip_lower converts to 'A'.

               # Which is correct.

               # So, the function compute_original(c, flip_upper, flip_lower) would be:

               # if flip_lower is True and c is uppercase:

               # m = c.lower()

               # else:

               # m = c

               # Then, compute o based on m and flip_upper.

               # So, for m and flip_upper:

               # if flip_upper is True:

               # o can be m (lowercase) or uppercase M.

               # But to ensure that applying flip_upper to o gives m.

               # So, if m is lowercase:

               # o can be m (lowercase) → flip_upper leaves it as m.

               # or o can be uppercase M (where M.lower() == m) → flip_upper converts to m.

               # So, choose o = m (lowercase).

               # So, the function would return m.

               # So, in code:

               m = c.lower()

               if flip_upper:

                   # o must be such that flip_upper applied to o gives m.

                   # o can be m (lowercase) or uppercase.

                   # choose o = m.

                   return m

               else:

                   # o is m.

                   return m

           else:

               # c is lowercase.

               # But flip_lower is True, which converts lowercase to uppercase.

               # So, m must be lowercase, and applying flip_lower to m gives c (lowercase).

               # But flip_lower converts lowercase to uppercase. So, m must be lowercase, and applying flip_lower gives uppercase.

               # So, c must be uppercase, which is a contradiction.

               # So, this case is impossible.

               # But according to the problem statement, it is allowed.

               # So, this suggests that the model is incorrect.

               # This indicates that the approach is not working.

               # This is getting too complicated.

               # Perhaps the correct approach is to represent the current state with two flags and apply the transformations on the fly, and for type 1 operations, compute the original character by applying the inverse transformations.

               # But given the time constraints, perhaps the intended solution is to use two flags and apply the transformations on the fly for each character when reading or writing.

               # So, the code would be:

               # - Read the original string into a list.

               # - Track two flags: flip_upper and flip_lower.

               # - For each operation:

               #   - If type 1: compute the current character, modify it, and store the original character after applying the inverse transformations.

               #   - If type 2: toggle flip_upper.

               #   - If type 3: toggle flip_lower.

               # Then, after all operations, apply the transformations to each character in the original list and output the result.

               # But this would require O(N) time for the final output, which is acceptable.

               # But for type 1 operations, each operation would take O(1) time, but requires applying the transformations to compute the current character, then applying the inverse transformations to store the new original character.

               # So, let's proceed with this approach.

               # So, the steps are:

               # 1. Read the original string into a list.

               # 2. Initialize flip_upper = False, flip_lower = False.

               # 3. For each operation:

               #    a. If type 1:

               #        i. Compute the current character at position x-1.

               #        ii. Apply the current transformations to the original character to get the current character.

               #        iii. Set the new current character to c.

               #        iv. Compute the new original character by applying the inverse transformations to c.

               #        v. Update the original list with the new original character.

               #    b. If type 2: toggle flip_upper.

               #    c. If type 3: toggle flip_lower.

               # 4. After all operations, apply the current transformations to each original character to get the final string.

               # But how to compute the current character from the original character and the flags.

               # Let's write a helper function to compute the current character:

               def get_current_char(o):

                   current = o

                   if flip_upper:

                       current = current.lower() if current.isupper() else current

                   if flip_lower:

                       current = current.upper() if current.islower() else current

                   return current

               # For a type 1 operation, we need to set the current character to c.

               # So, the original character o must be computed such that get_current_char(o) == c.

               # But how to compute o.

               # Let's write a helper function to compute o given c and the current flags.

               def compute_original(c):

                   # Apply the inverse transformations in reverse order.

                   # The current transformations are flip_upper followed by flip_lower.

                   # So, the inverse is to apply the inverse of flip_lower, then the inverse of flip_upper.

                   # Inverse of flip_lower: during the transformation, lowercase is converted to uppercase.

                   # So, to invert this, we need to convert uppercase to lowercase if flip_lower is True.

                   # Inverse of flip_upper: during transformation, uppercase is converted to lowercase.

                   # So, to invert this, we need to convert lowercase to uppercase if flip_upper is True.

                   # So, the steps are:

                   # 1. Apply inverse of flip_lower:

                   #    if flip_lower is True:

                   #        if c is uppercase: convert to lowercase.

                   #        else: remain as is.

                   # 2. Apply inverse of flip_upper:

                   #    if flip_upper is True:

                   #        if the result is lowercase: convert to uppercase.

                   #        else: remain as is.

                   # Wait, this may not be correct.

                   # Let's think:

                   # The transformation is:

                   # o -> flip_upper -> flip_lower -> c.

                   # To find o, we need to invert the steps.

                   # So, starting from c:

                   # after flip_lower is applied, the result was m.

                   # after flip_upper is applied, the result was o.

                   # So, m = flip_lower's inverse applied to c.

                   # o = flip_upper's inverse applied to m.

                   # So, what is the inverse of flip_lower?

                   # During flip_lower, lowercase is converted to uppercase.

                   # So, the inverse would be to convert uppercase to lowercase.

                   # But only if flip_lower is True.

                   # So, for the inverse of flip_lower:

                   # if flip_lower is True:

                   #    if c is uppercase: it was converted from lowercase, so m = c.lower()

                   #    else: m = c (since it wasn't converted).

                   # But this is not necessarily the case.

                   # For example, during flip_lower, if the character was uppercase, it remains uppercase.

                   # So, the inverse is not straightforward.

                   # This approach seems to be leading to a dead end.

                   # Given the time constraints, perhaps the correct solution is to proceed with the initial approach of using two flags and applying the transformations on the fly, and for type 1 operations, compute the original character by applying the inverse transformations.

                   # However, this may not handle all cases correctly, but given the problem's sample inputs and the time constraints, this may be the intended solution.

                   # So, here's the code:

                   # Read the input.

                   # Store the original characters in a list.

                   # Track flip_upper and flip_lower.

                   # For each type 1 operation:

                   #   compute the current character by applying the transformations to the original character.

                   #   set the new current character to c.

                   #   compute the new original character by applying the inverse transformations to c.

                   #   update the original list.

                   # For type 2 and 3, toggle the flags.

                   # Finally, apply the transformations to all original characters and output.

                   # But how to compute the inverse transformations.

                   # Let's proceed with the following code:

                   # For type 1 operation:

                   # x is 1-based index.

                   # original_char = original[x-1]

                   # current_char = apply_transformations(original_char)

                   # desired current_char is c.

                   # So, compute new_original_char = apply_inverse_transformations(c)

                   # original[x-1] = new_original_char