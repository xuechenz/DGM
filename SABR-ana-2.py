results = []
for alpha in alpha_list:
    abs_err_list = []
    pct_err_list = []
    for F0 in F_vals:
        sigma = hagan_implied_vol(F0, K, T, alpha, beta, rho, nu)
        analytic = bs_call(F0, K, T, sigma).item()
        pred = deep_price(F0, alpha)
        abs_err = abs(pred - analytic)
        abs_err_list.append(abs_err)

        # 只在解析价格大于阈值时计算百分比误差
        if analytic > 1e-1:
            pct_err = abs_err / analytic
            pct_err_list.append(pct_err)

    results.append({
        "alpha": alpha,
        "MAE": np.mean(abs_err_list),
        "Max Abs Err": np.max(abs_err_list),
        "Avg % Err": (np.mean(pct_err_list) * 100) if pct_err_list else None
    })

import pandas as pd
df = pd.DataFrame(results).round(6)
print(df)
