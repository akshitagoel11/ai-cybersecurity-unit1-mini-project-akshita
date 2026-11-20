"""Generate a synthetic login / network-summary dataset (no real user data).
Labels come from a risk score built from known threat indicators plus noise."""
import numpy as np, pandas as pd

SEED = 231
rng = np.random.default_rng(SEED)
N = 5000

countries = ["IN", "US", "UK", "DE", "FR", "RU", "CN", "UNKNOWN"]
c_prob    = [0.38, 0.18, 0.10, 0.08, 0.06, 0.07, 0.08, 0.05]
protocols = ["HTTPS", "HTTP", "SSH", "FTP", "RDP", "DNS"]
p_prob    = [0.40, 0.18, 0.12, 0.08, 0.10, 0.12]
devices   = ["laptop", "desktop", "mobile", "server", "iot"]

df = pd.DataFrame({
    "protocol": rng.choice(protocols, N, p=p_prob),
    "source_country": rng.choice(countries, N, p=c_prob),
    "device_type": rng.choice(devices, N, p=[.35, .25, .25, .08, .07]),
    "login_hour": np.clip(rng.normal(13, 5, N), 0, 23).astype(int),
    "failed_logins": rng.poisson(0.8, N),
    "session_duration_s": np.clip(rng.gamma(2.2, 180, N), 5, 3000).round(1),
    "bytes_in": rng.lognormal(8.5, 1.0, N).round(0),
    "bytes_out": rng.lognormal(8.3, 1.1, N).round(0),
    "ports_accessed": np.clip(rng.poisson(3, N) + 1, 1, 15),
    "password_age_days": rng.integers(1, 400, N),
    "mfa_enabled": rng.choice([0, 1], N, p=[0.3, 0.7]),
})

risk = np.zeros(N)
risk += 1.6 * df.source_country.isin(["RU", "CN", "UNKNOWN"])
risk += 0.9 * (df.protocol.isin(["RDP", "SSH", "FTP"]))
risk += 0.45 * df.failed_logins.clip(0, 6)
risk += 0.9 * ((df.login_hour < 5) | (df.login_hour > 22))
risk += 0.9 * (df.bytes_out > df.bytes_out.quantile(0.9))
risk += 0.25 * df.ports_accessed.clip(0, 10) * (df.ports_accessed > 6)
risk += 1.0 * (df.mfa_enabled == 0)
risk += 0.5 * (df.password_age_days > 300)
risk += rng.normal(0, 0.7, N)
thr = np.quantile(risk, 0.82)               # ~18% suspicious
df["is_suspicious"] = (risk > thr).astype(int)

# ~3% missing values in four columns (to practise preprocessing)
for col in ["login_hour", "bytes_in", "device_type", "session_duration_s"]:
    df.loc[rng.random(N) < 0.03, col] = np.nan

df.to_csv("data/cyber_events.csv", index=False)
print(df.shape, "suspicious:", int(df.is_suspicious.sum()))
