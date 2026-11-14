# Quantum_frenkel_langlands_sduality_regressor 



import random
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate, UnitaryGate
from qiskit.quantum_info import Operator, Statevector

random.seed(39)
np.random.seed(39)



FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90_loto_2981.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90_loto_plus_1737.csv"),
)



NUM_QUBITS = 6
STATE_DIM = 2 ** NUM_QUBITS
MAX_NUM = 39
NUM_POSITIONS = 7
HECKE_PRIMES = (2, 3, 5, 7, 11)
COUPLING_G_GRID = (0.45, 0.85, 1.35)
REG_GRID = (1e-3, 1e-2)


def load_draws(file_path: Path) -> np.ndarray:
    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean:
                continue
            parts = [p.strip() for p in clean.split(",") if p.strip()]
            nums = [int(x) for x in parts[:NUM_POSITIONS]]
            if len(nums) != NUM_POSITIONS:
                raise ValueError(f"Neispravan broj elemenata u redu: {clean}")
            if len(set(nums)) != NUM_POSITIONS or any(n < 1 or n > MAX_NUM for n in nums):
                raise ValueError(f"Neispravni brojevi u redu: {clean}")
            rows.append(sorted(nums))
    data = np.array(rows, dtype=np.int64)
    if len(data) < 40:
        raise ValueError(f"Nedovoljno istorijskih kola u fajlu: {file_path}")
    return data


def get_position_mask() -> np.ndarray:
    num_grid = np.arange(1, MAX_NUM + 1, dtype=np.int64)[None, :]
    pos_idx = np.arange(NUM_POSITIONS, dtype=np.int64)[:, None]
    min_allowed = pos_idx + 1
    max_allowed = MAX_NUM - (NUM_POSITIONS - 1 - pos_idx)
    return (num_grid >= min_allowed) & (num_grid <= max_allowed)






def build_hecke_operators_z39() -> list[np.ndarray]:
    def wrap_z39(val: int) -> int:
        return ((val - 1) % MAX_NUM) + 1

    hecke_list: list[np.ndarray] = []
    for p in HECKE_PRIMES:
        t_mat = np.zeros((MAX_NUM, MAX_NUM), dtype=np.float64)
        for x in range(1, MAX_NUM + 1):
            i = x - 1
            raw_prod = p * x
            y_mul = wrap_z39(raw_prod + 1) if (raw_prod % MAX_NUM == 0) else wrap_z39(raw_prod)
            j = y_mul - 1
            t_mat[i, j] += 0.5
            t_mat[j, i] += 0.5
        norm_t = float(np.linalg.norm(t_mat, ord=2))
        hecke_list.append(t_mat / max(norm_t, 1e-12))
    return hecke_list






def build_qft_operators() -> tuple[np.ndarray, np.ndarray]:
    qc = QuantumCircuit(NUM_QUBITS)
    qc.append(QFTGate(NUM_QUBITS), list(range(NUM_QUBITS)))
    u_qft = Operator(qc).data
    u_iqft = np.conj(u_qft.T)
    return u_qft, u_iqft


def build_langlands_sdual_unitaries(
    hecke_ops: list[np.ndarray],
    coupling_g: float,
    u_qft: np.ndarray,
    u_iqft: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h_aut = np.zeros((STATE_DIM, STATE_DIM), dtype=np.complex128)
    h_spec = np.zeros((STATE_DIM, STATE_DIM), dtype=np.complex128)

    n_primes = len(hecke_ops)
    for idx, t_p in enumerate(hecke_ops):
        w_aut = 1.0 / float(idx + 1)
        w_spec = 1.0 / float(n_primes - idx)
        h_aut[:MAX_NUM, :MAX_NUM] += w_aut * t_p
        h_spec[:MAX_NUM, :MAX_NUM] += w_spec * t_p

    for k in range(MAX_NUM, STATE_DIM):
        buffer_e = 1.5 + 0.1 * float(k - MAX_NUM)
        h_aut[k, k] = buffer_e
        h_spec[k, k] = buffer_e

    h_aut = 0.5 * (h_aut + np.conj(h_aut.T))
    h_spec = 0.5 * (h_spec + np.conj(h_spec.T))

    u_aut_raw = expm(-1j * coupling_g * h_aut)
    u_a, _, vh_a = np.linalg.svd(u_aut_raw)
    u_aut = u_a @ vh_a

    dual_g = 1.0 / max(coupling_g, 1e-4)
    u_spec_raw = expm(-1j * dual_g * h_spec)
    u_spec = u_iqft @ u_spec_raw @ u_qft
    u_s, _, vh_s = np.linalg.svd(u_spec)
    u_spec = u_s @ vh_s
    return u_aut, u_spec


def precompute_draw_amplitudes(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_total = len(draws)
    n_samples = n_total - 1
    phases = np.exp(1j * np.arange(NUM_POSITIONS, dtype=np.float64) * np.pi / float(NUM_POSITIONS))
    norm_f = float(np.sqrt(NUM_POSITIONS))

    init_amps = np.zeros((n_samples, STATE_DIM), dtype=np.complex128)
    y_targets = np.empty((n_samples, NUM_POSITIONS), dtype=np.int64)

    for idx in range(n_samples):
        prev_zero = draws.__getitem__(idx) - 1
        init_amps[idx, prev_zero] = phases / norm_f
        y_targets[idx] = draws.__getitem__(idx + 1)

    latest_amp = np.zeros(STATE_DIM, dtype=np.complex128)
    latest_zero = draws.__getitem__(n_total - 1) - 1
    latest_amp[latest_zero] = phases / norm_f
    return init_amps, y_targets, latest_amp


def extract_sdual_densities(
    init_amps: np.ndarray,
    u_aut: np.ndarray,
    u_spec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    psi_aut = np.einsum("ij,nj->ni", u_aut, init_amps)
    psi_spec = np.einsum("ij,nj->ni", u_spec, init_amps)
    dens_aut = np.abs(psi_aut[:, :MAX_NUM]) ** 2
    dens_spec = np.abs(psi_spec[:, :MAX_NUM]) ** 2
    return dens_aut, dens_spec


def extract_latest_sdual_via_qiskit(
    latest_amp: np.ndarray,
    u_aut: np.ndarray,
    u_spec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    qubits = list(range(NUM_QUBITS))

    qc_aut = QuantumCircuit(NUM_QUBITS)
    qc_aut.append(UnitaryGate(u_aut, label="LanglandsAut"), qubits)
    sv_aut = Statevector(latest_amp).evolve(qc_aut).data

    qc_spec = QuantumCircuit(NUM_QUBITS)
    qc_spec.append(UnitaryGate(u_spec, label="LanglandsSpec"), qubits)
    sv_spec = Statevector(latest_amp).evolve(qc_spec).data

    dens_aut = (np.abs(sv_aut[:MAX_NUM]) ** 2)[None, :]
    dens_spec = (np.abs(sv_spec[:MAX_NUM]) ** 2)[None, :]
    return dens_aut, dens_spec


def sdual_densities_to_probs(
    dens_aut: np.ndarray,
    dens_spec: np.ndarray,
    gain_aut: np.ndarray,
    gain_spec: np.ndarray,
    pos_bias: np.ndarray,
    pos_mask: np.ndarray,
) -> np.ndarray:
    log_a = np.log(np.maximum(dens_aut, 1e-12))
    log_s = np.log(np.maximum(dens_spec, 1e-12))

    logits = (
        gain_aut[None, :, None] * log_a[:, None, :]
        + gain_spec[None, :, None] * log_s[:, None, :]
        + pos_bias[None, :, :]
    )
    logits_masked = np.where(pos_mask[None, :, :], logits, -1e9)
    logits_shifted = logits_masked - np.max(logits_masked, axis=2, keepdims=True)
    exp_l = np.where(pos_mask[None, :, :], np.exp(np.clip(logits_shifted, -30.0, 30.0)), 0.0)
    denom = np.sum(exp_l, axis=2, keepdims=True)
    uniform_fb = pos_mask[None, :, :].astype(np.float64) / np.sum(pos_mask, axis=1, keepdims=True)[None, :, :]
    return np.where(denom > 1e-15, exp_l / np.maximum(denom, 1e-15), uniform_fb)


def fit_langlands_sduality_regressor(
    aut_train: np.ndarray,
    spec_train: np.ndarray,
    y_train: np.ndarray,
    pos_mask: np.ndarray,
    reg_weight: float,
    maxiter: int = 55,
) -> dict:
    n_samples = len(y_train)
    sample_idx = np.arange(n_samples)[:, None]
    pos_idx = np.arange(NUM_POSITIONS)[None, :]
    target_idx = y_train - 1

    one_hot = np.zeros((n_samples, NUM_POSITIONS, MAX_NUM), dtype=np.float64)
    one_hot[sample_idx, pos_idx, target_idx] = 1.0

    log_a = np.log(np.maximum(aut_train, 1e-12))
    log_s = np.log(np.maximum(spec_train, 1e-12))
    uniform_fb = pos_mask[None, :, :].astype(np.float64) / np.sum(pos_mask, axis=1, keepdims=True)[None, :, :]

    init_ga = np.full(NUM_POSITIONS, 0.5, dtype=np.float64)
    init_gs = np.full(NUM_POSITIONS, 0.5, dtype=np.float64)
    init_bias = np.zeros((NUM_POSITIONS, MAX_NUM), dtype=np.float64)
    init_params = np.concatenate([init_ga, init_gs, init_bias.reshape(-1)])
    n_total_params = float(len(init_params))

    def loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
        ga = params[:NUM_POSITIONS]
        gs = params[NUM_POSITIONS : 2 * NUM_POSITIONS]
        bias = params[2 * NUM_POSITIONS :].reshape(NUM_POSITIONS, MAX_NUM)

        logits = (
            ga[None, :, None] * log_a[:, None, :]
            + gs[None, :, None] * log_s[:, None, :]
            + bias[None, :, :]
        )
        logits_masked = np.where(pos_mask[None, :, :], logits, -1e9)
        logits_shifted = logits_masked - np.max(logits_masked, axis=2, keepdims=True)
        exp_l = np.where(pos_mask[None, :, :], np.exp(np.clip(logits_shifted, -30.0, 30.0)), 0.0)
        denom = np.sum(exp_l, axis=2, keepdims=True)
        probs = np.where(denom > 1e-15, exp_l / np.maximum(denom, 1e-15), uniform_fb)

        chosen = probs[sample_idx, pos_idx, target_idx]
        nll = -float(np.mean(np.sum(np.log(np.maximum(chosen, 1e-12)), axis=1)))
        reg = reg_weight * float(np.mean(params ** 2))
        total_loss = nll + reg

        d_logits = np.where(pos_mask[None, :, :], probs - one_hot, 0.0) / float(n_samples)
        grad_ga = np.sum(d_logits * log_a[:, None, :], axis=(0, 2)) + (2.0 * reg_weight / n_total_params) * ga
        grad_gs = np.sum(d_logits * log_s[:, None, :], axis=(0, 2)) + (2.0 * reg_weight / n_total_params) * gs
        grad_bias = np.sum(d_logits, axis=0) + (2.0 * reg_weight / n_total_params) * bias

        return total_loss, np.concatenate([grad_ga, grad_gs, grad_bias.reshape(-1)])

    res = minimize(
        loss_and_grad,
        init_params,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "ftol": 1e-7, "gtol": 1e-5},
    )

    opt = res.x
    return {
        "gain_aut": opt[:NUM_POSITIONS],
        "gain_spec": opt[NUM_POSITIONS : 2 * NUM_POSITIONS],
        "pos_bias": opt[2 * NUM_POSITIONS :].reshape(NUM_POSITIONS, MAX_NUM),
        "pos_mask": pos_mask,
    }





def evaluate_langlands_nll(model: dict, aut_eval: np.ndarray, spec_eval: np.ndarray, y_eval: np.ndarray) -> float:
    probs = sdual_densities_to_probs(
        aut_eval,
        spec_eval,
        model.__getitem__("gain_aut"),
        model.__getitem__("gain_spec"),
        model.__getitem__("pos_bias"),
        model.__getitem__("pos_mask"),
    )
    n_samples = len(y_eval)
    sample_idx = np.arange(n_samples)[:, None]
    pos_idx = np.arange(NUM_POSITIONS)[None, :]
    target_idx = y_eval - 1
    chosen = probs[sample_idx, pos_idx, target_idx]
    return -float(np.mean(np.sum(np.log(np.maximum(chosen, 1e-12)), axis=1)))


def decode_hecke_eigenform_chain(pos_probs: np.ndarray, pos_mask: np.ndarray) -> list[int]:
    neg_inf = -1e18
    dp = np.full((NUM_POSITIONS, MAX_NUM), neg_inf, dtype=np.float64)
    parent = np.full((NUM_POSITIONS, MAX_NUM), -1, dtype=np.int64)

    first_mask = pos_mask.__getitem__(0)
    first_p = pos_probs.__getitem__(0)
    for k in range(MAX_NUM):
        if bool(first_mask.__getitem__(k)):
            dp[0, k] = np.log(max(float(first_p.__getitem__(k)), 1e-12))

    for pos in range(1, NUM_POSITIONS):
        curr_mask = pos_mask.__getitem__(pos)
        curr_p = pos_probs.__getitem__(pos)
        prev_dp = dp.__getitem__(pos - 1)
        for k in range(MAX_NUM):
            if not bool(curr_mask.__getitem__(k)):
                continue
            best_prev = neg_inf
            best_idx = -1
            for prev_k in range(k):
                val = float(prev_dp.__getitem__(prev_k))
                if val > best_prev:
                    best_prev = val
                    best_idx = prev_k
            if best_idx >= 0 and best_prev > neg_inf * 0.5:
                dp[pos, k] = best_prev + np.log(max(float(curr_p.__getitem__(k)), 1e-12))
                parent[pos, k] = best_idx

    last_dp = dp.__getitem__(NUM_POSITIONS - 1)
    best_last = int(np.argmax(last_dp))
    if float(last_dp.__getitem__(best_last)) <= neg_inf * 0.5:
        raise RuntimeError("Neuspesno Hecke S-dualno dekodiranje 7 razlicitih brojeva.")

    seq_rev: list[int] = []
    curr_k = best_last
    for pos in range(NUM_POSITIONS - 1, -1, -1):
        seq_rev.append(curr_k + 1)
        curr_k = int(parent[pos, curr_k])

    seq = list(reversed(seq_rev))
    if len(set(seq)) != NUM_POSITIONS:
        raise RuntimeError("Dekodirana kombinacija nema 7 razlicitih brojeva.")
    return seq


def main() -> None:
    print("Metoda: Edward Frenkel Geometric Langlands S-Duality & Hecke Regressor (1x6 qubita)")
    print("Posebnost: Modularni Hecke operatori T_p na Z_39 + S-dualnost (g <-> 1/g) na Qiskit simulatoru")

    pos_mask = get_position_mask()
    hecke_ops = build_hecke_operators_z39()
    u_qft, u_iqft = build_qft_operators()

    for file_path in FILES:
        if not file_path.exists():
            print(f"Fajl nije pronadjen: {file_path}")
            continue

        draws = load_draws(file_path)
        init_amps, y_all, latest_amp = precompute_draw_amplitudes(draws)

        n_samples = int(y_all.shape.__getitem__(0))
        train_end = int(0.75 * n_samples)
        val_end = int(0.88 * n_samples)

        best_g = float(COUPLING_G_GRID.__getitem__(0))
        best_reg = float(REG_GRID.__getitem__(0))
        best_val_nll = float("inf")
        cached_sdual: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

        for g_val in COUPLING_G_GRID:
            u_aut, u_spec = build_langlands_sdual_unitaries(hecke_ops, g_val, u_qft, u_iqft)
            dens_aut, dens_spec = extract_sdual_densities(init_amps, u_aut, u_spec)
            cached_sdual[g_val] = (dens_aut, dens_spec, u_aut, u_spec)

            for reg_w in REG_GRID:
                model = fit_langlands_sduality_regressor(
                    dens_aut[:train_end],
                    dens_spec[:train_end],
                    y_all[:train_end],
                    pos_mask,
                    reg_weight=reg_w,
                    maxiter=55,
                )
                val_nll = evaluate_langlands_nll(
                    model,
                    dens_aut[train_end:val_end],
                    dens_spec[train_end:val_end],
                    y_all[train_end:val_end],
                )
                if val_nll < best_val_nll:
                    best_val_nll = val_nll
                    best_g = g_val
                    best_reg = reg_w

        aut_best, spec_best, u_aut_best, u_spec_best = cached_sdual.__getitem__(best_g)
        eval_model = fit_langlands_sduality_regressor(
            aut_best[:train_end],
            spec_best[:train_end],
            y_all[:train_end],
            pos_mask,
            reg_weight=best_reg,
            maxiter=55,
        )
        test_nll = evaluate_langlands_nll(eval_model, aut_best[val_end:], spec_best[val_end:], y_all[val_end:])

        final_model = fit_langlands_sduality_regressor(
            aut_best,
            spec_best,
            y_all,
            pos_mask,
            reg_weight=best_reg,
            maxiter=55,
        )
        lat_aut, lat_spec = extract_latest_sdual_via_qiskit(latest_amp, u_aut_best, u_spec_best)
        next_probs = sdual_densities_to_probs(
            lat_aut,
            lat_spec,
            final_model.__getitem__("gain_aut"),
            final_model.__getitem__("gain_spec"),
            final_model.__getitem__("pos_bias"),
            pos_mask,
        ).__getitem__(0)
        next_combo = decode_hecke_eigenform_chain(next_probs, pos_mask)

        print(f"CSV: {file_path.name}")
        print(f"Validacija NLL (g={best_g}, reg={best_reg}): {best_val_nll:.4f}")
        print(f"Test NLL: {test_nll:.4f}")
        print(f"NEXT: {next_combo}")


if __name__ == "__main__":
    main()



"""
Metoda: Edward Frenkel Geometric Langlands S-Duality & Hecke Regressor (1x6 qubita)
Posebnost: Modularni Hecke operatori T_p na Z_39 + S-dualnost (g <-> 1/g) na Qiskit simulatoru

CSV: loto7_4718_k90.csv
Validacija NLL (g=1.35, reg=0.001): 20.4833
Test NLL: 20.4327
NEXT: [1, 8, 13, 18, 29, 33, 39]

CSV: loto7_4718_k90_loto_2981.csv
Validacija NLL (g=1.35, reg=0.001): 20.4942
Test NLL: 20.5963
NEXT: [1, 8, 13, 21, 29, 32, 39]

CSV: loto7_4718_k90_loto_plus_1737.csv
Validacija NLL (g=1.35, reg=0.001): 20.2305
Test NLL: 20.3987
NEXT: [1, 8, 11, 18, 27, 34, 39]
"""



"""
Edward Frenkel

Oblast za Edwarda Frenkela odnosi se na Langlandsov program (posebno Geometrijski Langlandsov program i kvantnu dualnost / S-dualnost između spektralne i automorfne strane).

Oblast Langlandsovog programa (automorfne forme, spektralni Hecke operatori i S-dualnost / Furije–Mukai dualnost između reprezentacija i geometrije) do sada nije obrađena u ovom razgovoru.

• Osnovna ideja (Edward Frenkel — Langlandsov program i S-dualnost):
	• Langlandsov program povezuje dve naizgled potpuno različite oblasti matematike: Galoisove / spektralne reprezentacije (simetrije i sopstvene vrednosti operatora) i automorfne forme (harmonijske strukture i modularne funkcije nad aritmetičkim prostorima), dok u kvantnoj fizici (Kapustin-Witten / Frenkel) to odgovara elektromagnetnoj S-dualnosti (\(g \leftrightarrow 1/g\)), gde se jaki i slabi režim međusobno preslikavaju preko Furijeove / modularne transformacije.

	1. Automorfna strana — Hecke-ovi operatori \(\^{T}_{p}\) na \(\mathbb{Z}_{39}\): Za skup prostih delilaca/koraka \(p \in \{2, 3, 5, 7, 11\}\) u cikličnom prostoru od 39 brojeva definišemo Hecke-ove operatore \(\^{T}_{p}\) koji spajaju broj x sa njegovim modularnim slikama \((p \cdot x) \bmod 39\) i \((p^{-1} \cdot x) \bmod 39\).
	2. Spektralna strana i Qiskit S-dualna evolucija na 1x6 qubita: Kvantno kolo na 6 qubita na lokalnom simulatoru (QFTGate + unitarna Hecke evolucija) povezuje direktno stanje (automorfni režim sa konstantom veze g) i QFT-dualno stanje (spektralni režim sa dualnom konstantom 1/g), izdvajajući S-dualni balans spektralnih i automorfnih gustina po brojevima \(1 \dots 39\).
	3. Hecke-Eigenform & S-Dual Selection dekoder: NEXT kombinacija od 7 različitih brojeva dobija se iz S-dualne verovatnoćne raspodele po pozicijama \(1 \dots 7\) uz očuvanje strogog poretka \(1 \le x_1 < \dots < x_7 \le 39\).

Multiplikativna modularna simetriju (Hecke operatore \(x \mapsto p x \bmod 39\)) i S-dualnost umesto isključivo aditivnih pomeraja.
Hecke-ovi operatori otkrivaju multiplikativne kongruencije u \(\mathbb{Z}_{39}\), dok S-dualnost spaja originalni i Furije-dualni prikaz na 6 qubita.
"""
