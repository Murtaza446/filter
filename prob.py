# Root probabilities
p_t_true = 0.5
p_t_false = 0.5
p_r_true = 0.65
p_r_false = 0.35

# E and P as per previous definitions
def compute_p_p_given_r(r_val):
    if r_val:
        e_true_tt = 0.85
        e_true_ft = 0.55
        p = (
            p_t_true * (e_true_tt * p_p_t_e_t + (1 - e_true_tt) * p_p_t_e_f) +
            p_t_false * (e_true_ft * p_p_t_e_t + (1 - e_true_ft) * p_p_t_e_f)
        )
    else:
        e_true_tf = 0.4
        e_true_ff = 0.1
        p = (
            p_t_true * (e_true_tf * p_p_t_e_t + (1 - e_true_tf) * p_p_t_e_f) +
            p_t_false * (e_true_ff * p_p_t_e_t + (1 - e_true_ff) * p_p_t_e_f)
        )
    return p

p_p_r_true = compute_p_p_given_r(True)
p_p_r_false = compute_p_p_given_r(False)

# Total P(P=True)
p_p_true = p_p_r_true * p_r_true + p_p_r_false * p_r_false

# Bayes' Theorem
p_r_true_given_p = (p_p_r_true * p_r_true) / p_p_true

print("P(Review=True | Promotion=True) =", round(p_r_true_given_p, 4))