// =========================================
// 1. HELPER & UTILITY (Parsing & Math)
// =========================================

// Evaluator fungsi f(x)
function f(x, fx_string) {
    try {
        if (!fx_string) return 0;
        // Ganti euler 'e' menjadi Math.E, 'pi' jadi Math.PI
        let cleaned = fx_string.toLowerCase()
            .replace(/\^/g, '**')
            .replace(/\b(sin|cos|tan|log|exp|sqrt|abs|asin|acos|atan)\b/g, 'Math.$1')
            .replace(/\be\b/g, 'Math.E').replace(/\bpi\b/g, 'Math.PI');
        return new Function("x", "return " + cleaned)(x);
    } catch (e) { return NaN; }
}

// Evaluator f(x,y) untuk ODE
function fxy(x, y, expr) {
    try {
        let cleaned = expr.toLowerCase()
            .replace(/\^/g, '**')
            .replace(/\b(sin|cos|tan|log|exp|sqrt)\b/g, 'Math.$1');
        return new Function("x", "y", "return " + cleaned)(x, y);
    } catch (e) { return NaN; }
}

// Parsing Input CSV (1, 2, 3)
function parseCSV(id) {
    const el = document.getElementById(id);
    if (!el || !el.value.trim()) throw new Error(`Field ${id} kosong.`);
    return el.value.split(',').map(x => parseFloat(x.trim())).filter(x => !isNaN(x));
}

// Parsing Matrix (Baris enter, Kolom koma)
function parseMatrix(id) {
    const el = document.getElementById(id);
    if (!el || !el.value.trim()) throw new Error(`Matriks ${id} kosong.`);
    return el.value.trim().split('\n').map(row => 
        row.split(',').map(num => parseFloat(num.trim())).filter(n => !isNaN(n))
    );
}

// Membuat Tabel HTML
function createTable(headers, rows, info="") {
    let h = "";
    if(info) h += `<div class="alert alert-info shadow-sm mb-4 text-sm font-medium">${info}</div>`;
    h += `<div class="overflow-x-auto"><table class="table table-zebra table-xs md:table-sm w-full border text-center">
            <thead class="bg-primary text-white"><tr>`;
    headers.forEach(head => h += `<th>${head}</th>`);
    h += `</tr></thead><tbody>`;
    rows.forEach(row => {
        h += `<tr>`;
        row.forEach(cell => {
            let val = (typeof cell === 'number') ? parseFloat(cell.toFixed(6)) : cell;
            h += `<td class="font-mono">${val}</td>`;
        });
        h += `</tr>`;
    });
    h += `</tbody></table></div>`;
    return h;
}

// --- ENGINE LINEAR ALGEBRA MINI (Penting untuk Regresi Polinomial) ---
const LinearAlgebra = {
    // Menyelesaikan Ax = B menggunakan Gauss-Jordan
    solve: (A, B) => {
        let n = A.length;
        // Deep copy matriks agar input asli tidak rusak
        let M = A.map((row, i) => [...row, B[i]]); 

        for (let i = 0; i < n; i++) {
            // Pivoting
            let maxRow = i;
            for (let k = i + 1; k < n; k++) {
                if (Math.abs(M[k][i]) > Math.abs(M[maxRow][i])) maxRow = k;
            }
            [M[i], M[maxRow]] = [M[maxRow], M[i]];

            if (Math.abs(M[i][i]) < 1e-9) return null; // Singular

            // Normalisasi baris pivot
            let pivot = M[i][i];
            for (let j = i; j <= n; j++) M[i][j] /= pivot;

            // Eliminasi baris lain
            for (let k = 0; k < n; k++) {
                if (k !== i) {
                    let factor = M[k][i];
                    for (let j = i; j <= n; j++) M[k][j] -= factor * M[i][j];
                }
            }
        }
        return M.map(row => row[n]); // Kembalikan kolom terakhir (hasil x)
    },
    
    // Matriks Invers (Untuk Direct Method SPL)
    inverse: (A) => {
        let n = A.length;
        // Gabungkan A dengan Matriks Identitas
        let M = A.map((row, i) => [
            ...row, 
            ...Array.from({length: n}, (_, k) => (i === k ? 1 : 0))
        ]);

        for (let i = 0; i < n; i++) {
            let pivot = M[i][i];
            if(Math.abs(pivot) < 1e-9) throw new Error("Matriks Singular, tidak punya invers.");
            for(let j=0; j<2*n; j++) M[i][j] /= pivot;
            for(let k=0; k<n; k++) {
                if (k !== i) {
                    let f = M[k][i];
                    for(let j=0; j<2*n; j++) M[k][j] -= f * M[i][j];
                }
            }
        }
        return M.map(row => row.slice(n)); // Ambil sisi kanan
    }
};

// =========================================
// 2. SOLVER LOGIC (26 METODE LENGKAP)
// =========================================
const Solver = {
    // === PERSAMAAN NON-LINEAR ===
    bisection: (fs, x1, x2, tol, max) => {
        if (f(x1, fs) * f(x2, fs) >= 0) throw new Error("Syarat gagal: f(x1)*f(x2) harus < 0.");
        let rows = [];
        for (let i = 1; i <= max; i++) {
            let xr = (x1 + x2) / 2;
            let fxr = f(xr, fs);
            rows.push([i, x1, x2, xr, fxr]);
            if (Math.abs(fxr) < tol || Math.abs(x2-x1) < tol) break;
            if (f(x1, fs) * fxr < 0) x2 = xr; else x1 = xr;
        }
        return { h: ["Iter", "x1", "x2", "xr", "f(xr)"], r: rows };
    },
    regula: (fs, x1, x2, tol, max) => {
        if (f(x1, fs) * f(x2, fs) >= 0) throw new Error("Syarat gagal: f(x1)*f(x2) harus < 0.");
        let rows = [];
        for (let i = 1; i <= max; i++) {
            let fx1 = f(x1, fs), fx2 = f(x2, fs);
            let xr = x2 - (fx2 * (x2 - x1)) / (fx2 - fx1);
            let fxr = f(xr, fs);
            rows.push([i, x1, x2, xr, fxr]);
            if (Math.abs(fxr) < tol) break;
            if (fx1 * fxr < 0) x2 = xr; else x1 = xr;
        }
        return { h: ["Iter", "x1", "x2", "xr", "f(xr)"], r: rows };
    },
    direct_nonlinear: (fs, x0, tol, max) => {
        // Metode Iterasi Titik Tetap (Fixed Point): x = g(x)
        // User memasukkan g(x) pada input fungsi.
        // Contoh f(x) = x^2 - 2x - 3 = 0  => x = sqrt(2x + 3) atau x = (x^2-3)/2
        let rows = [];
        let x = x0;
        for(let i=1; i<=max; i++){
            let x_next = f(x, fs); // Di sini fungsi input dianggap sebagai g(x)
            let err = Math.abs(x_next - x);
            rows.push([i, x, x_next, err]);
            x = x_next;
            if(err < tol) break;
        }
        return { 
            h: ["Iter", "x_i", "x_i+1 (g(x))", "Error"], 
            r: rows, 
            info: "Pastikan input fungsi adalah bentuk g(x), bukan f(x)=0." 
        };
    },
    newton: (fs, dfs, x0, tol, max) => {
        let rows = [];
        let x = x0;
        for(let i=1; i<=max; i++){
            let fx = f(x, fs);
            let dfx = f(x, dfs);
            if(Math.abs(dfx) < 1e-9) throw new Error("Turunan 0.");
            let xnew = x - fx/dfx;
            rows.push([i, x, fx, dfx, xnew]);
            if(Math.abs(xnew-x) < tol) break;
            x = xnew;
        }
        return { h: ["Iter", "xi", "f(xi)", "f'(xi)", "xi+1"], r: rows };
    },
    secant: (fs, x1, x2, tol, max) => {
        let rows = [];
        let x_prev = x1, x_curr = x2;
        for(let i=1; i<=max; i++){
            let fx_prev = f(x_prev, fs);
            let fx_curr = f(x_curr, fs);
            if(Math.abs(fx_curr - fx_prev) < 1e-12) break;
            let x_next = x_curr - (fx_curr * (x_curr - x_prev)) / (fx_curr - fx_prev);
            rows.push([i, x_prev, x_curr, x_next, Math.abs(x_next-x_curr)]);
            x_prev = x_curr; x_curr = x_next;
            if(Math.abs(rows[i-1][4]) < tol) break;
        }
        return { h: ["Iter", "x(i-1)", "x(i)", "x(i+1)", "Error"], r: rows };
    },

    // === SISTEM PERSAMAAN LINEAR ===
    gauss: (A, Bvec) => {
        // Gabung A dan B
        let n = A.length;
        let M = A.map((row, i) => [...row, Bvec[i][0]]);
        
        // Eliminasi Maju
        for(let k=0; k<n; k++){
            if(Math.abs(M[k][k]) < 1e-9) { /* logic swap baris sederhana */ 
                let idx = M.slice(k).findIndex(r => Math.abs(r[k]) > 1e-9);
                if(idx !== -1) [M[k], M[k+idx]] = [M[k+idx], M[k]];
            }
            for(let i=k+1; i<n; i++){
                let factor = M[i][k]/M[k][k];
                for(let j=k; j<=n; j++) M[i][j] -= factor * M[k][j];
            }
        }
        // Substitusi Mundur
        let X = Array(n).fill(0);
        for(let i=n-1; i>=0; i--){
            let sum = 0;
            for(let j=i+1; j<n; j++) sum += M[i][j]*X[j];
            X[i] = (M[i][n] - sum)/M[i][i];
        }
        return { h: ["Variabel", "Nilai"], r: X.map((v,i) => [`x${i+1}`, v]) };
    },
    gauss_jordan: (A, Bvec) => {
        let n = A.length;
        let M = A.map((row, i) => [...row, Bvec[i][0]]);
        for(let i=0; i<n; i++){
            let pivot = M[i][i];
            for(let j=0; j<=n; j++) M[i][j] /= pivot;
            for(let k=0; k<n; k++){
                if(k!==i){
                    let f = M[k][i];
                    for(let j=0; j<=n; j++) M[k][j] -= f*M[i][j];
                }
            }
        }
        return { h: ["Variabel", "Nilai"], r: M.map((row,i) => [`x${i+1}`, row[n]]) };
    },
    jacobi: (A, Bvec, init, tol, max) => {
        let n = A.length;
        let x = [...init];
        let rows = [];
        for(let k=1; k<=max; k++){
            let x_new = [];
            for(let i=0; i<n; i++){
                let sum = 0;
                for(let j=0; j<n; j++) if(j!==i) sum += A[i][j]*x[j];
                x_new[i] = (Bvec[i][0] - sum)/A[i][i];
            }
            let err = Math.sqrt(x_new.reduce((acc, val, idx) => acc + (val-x[idx])**2, 0));
            rows.push([k, ...x_new, err]);
            x = x_new;
            if(err < tol) break;
        }
        return { h: ["Iter", ...A.map((_,i)=>`x${i+1}`), "Error"], r: rows };
    },
    gauss_seidel: (A, Bvec, init, tol, max) => {
        let n = A.length;
        let x = [...init];
        let rows = [];
        for(let k=1; k<=max; k++){
            let x_old = [...x];
            for(let i=0; i<n; i++){
                let sum = 0;
                for(let j=0; j<n; j++){
                    if(j!==i) sum += A[i][j]*x[j]; // Gunakan nilai x terbaru jika tersedia
                }
                x[i] = (Bvec[i][0] - sum)/A[i][i];
            }
            let err = Math.sqrt(x.reduce((acc, val, idx) => acc + (val-x_old[idx])**2, 0));
            rows.push([k, ...x, err]);
            if(err < tol) break;
        }
        return { h: ["Iter", ...A.map((_,i)=>`x${i+1}`), "Error"], r: rows };
    },
    direct_spl: (A, Bvec) => {
        // Metode Invers Matriks: X = A^-1 * B
        try {
            let A_inv = LinearAlgebra.inverse(A);
            let n = A.length;
            let X = Array(n).fill(0);
            // Perkalian Matriks A_inv * Bvec
            for(let i=0; i<n; i++){
                for(let k=0; k<n; k++){
                    X[i] += A_inv[i][k] * Bvec[k][0];
                }
            }
            return { h: ["Var", "Nilai"], r: X.map((v,i) => [`x${i+1}`, v]), info: "Menggunakan Metode Invers Matriks (X = A⁻¹B)" };
        } catch(e) { throw new Error("Gagal menghitung invers (Matriks mungkin Singular)."); }
    },

    // === REGRESI ===
    reg_linier: (X, Y) => {
        let n = X.length;
        let sX=0, sY=0, sXY=0, sX2=0;
        X.forEach((x, i) => { sX+=x; sY+=Y[i]; sXY+=x*Y[i]; sX2+=x*x; });
        let a1 = (n*sXY - sX*sY)/(n*sX2 - sX*sX);
        let a0 = (sY/n) - a1*(sX/n);
        let rows = X.map((x,i) => [i+1, x, Y[i], (a0 + a1*x)]);
        return { h:["No","X","Y_act","Y_pred"], r:rows, info: `Model: Y = ${a0.toFixed(5)} + ${a1.toFixed(5)}x` };
    },
    reg_nonlinier: (X, Y) => {
        // Model Eksponensial: y = ae^(bx) -> ln y = ln a + bx
        // Transformasi: Y' = ln(y), A0 = ln(a), A1 = b
        let n = X.length;
        let sX=0, sY_log=0, sXY_log=0, sX2=0;
        let rows = [];
        try {
            for(let i=0; i<n; i++){
                if(Y[i] <= 0) throw new Error("Y harus > 0 untuk regresi eksponensial.");
                let logY = Math.log(Y[i]);
                sX += X[i]; sY_log += logY;
                sXY_log += X[i]*logY; sX2 += X[i]*X[i];
            }
            let b = (n*sXY_log - sX*sY_log)/(n*sX2 - sX*sX);
            let ln_a = (sY_log/n) - b*(sX/n);
            let a = Math.exp(ln_a);
            
            rows = X.map((x,i) => [x, Y[i], a * Math.exp(b*x)]);
            return { h:["X","Y_act","Y_model"], r:rows, info: `Model Eksponensial: Y = ${a.toFixed(5)} * e^(${b.toFixed(5)}x)` };
        } catch(e) { throw e; }
    },
    reg_polinomial: (X, Y, orde) => {
        let m = parseInt(orde);
        let n = X.length;
        if(n < m+1) throw new Error("Jumlah data kurang untuk orde ini.");
        
        // Membangun matriks Normal (Sigma x^pow)
        let A = Array.from({length: m+1}, () => Array(m+1).fill(0));
        let B = Array(m+1).fill(0);
        
        for(let i=0; i<=m; i++){
            for(let j=0; j<=m; j++){
                let sumPow = X.reduce((acc, val) => acc + Math.pow(val, i+j), 0);
                A[i][j] = sumPow;
            }
            B[i] = X.reduce((acc, val, idx) => acc + (Y[idx] * Math.pow(val, i)), 0);
        }
        
        // Selesaikan SPL untuk mencari koefisien a0, a1, a2...
        let coeffs = LinearAlgebra.solve(A, B);
        if(!coeffs) throw new Error("Gagal menghitung koefisien (Singular).");
        
        let formula = "Y = " + coeffs.map((c, i) => `${c.toFixed(4)}x^${i}`).join(" + ");
        return { h: ["Koefisien", "Nilai"], r: coeffs.map((c,i) => [`a${i}`, c]), info: formula };
    },

    // === INTERPOLASI ===
    inter_linier: (X, Y, xt) => {
        let i = 0; while(i < X.length-1 && xt > X[i+1]) i++; // Cari interval
        let y_res = Y[i] + ((Y[i+1]-Y[i])/(X[i+1]-X[i])) * (xt - X[i]);
        return { h:["Titik 1", "Titik 2", "x target", "Hasil y"], r:[[`(${X[i]},${Y[i]})`, `(${X[i+1]},${Y[i+1]})`, xt, y_res]] };
    },
    inter_kuadrat: (X, Y, xt) => {
        // Butuh 3 titik. Cari titik terdekat dengan xt.
        let n = X.length;
        if(n < 3) throw new Error("Interpolasi Kuadrat butuh minimal 3 titik.");
        
        // Cari idx dimana X[idx] paling dekat dengan xt
        let idx = 0;
        let minDiff = Math.abs(xt - X[0]);
        for(let i=1; i<n; i++){
            if(Math.abs(xt - X[i]) < minDiff) { minDiff = Math.abs(xt - X[i]); idx = i; }
        }
        
        // Ambil 3 titik sekitar (tangani batas array)
        let start = idx - 1;
        if(start < 0) start = 0;
        if(start + 2 >= n) start = n - 3;
        
        let x0=X[start], x1=X[start+1], x2=X[start+2];
        let y0=Y[start], y1=Y[start+1], y2=Y[start+2];
        
        // Rumus Lagrange Orde 2 (Kuadrat)
        let L0 = ((xt-x1)*(xt-x2)) / ((x0-x1)*(x0-x2));
        let L1 = ((xt-x0)*(xt-x2)) / ((x1-x0)*(x1-x2));
        let L2 = ((xt-x0)*(xt-x1)) / ((x2-x0)*(x2-x1));
        
        let res = y0*L0 + y1*L1 + y2*L2;
        return { h:["Points Used", "x target", "Result"], r:[ [`(${x0},${y0}), (${x1},${y1}), (${x2},${y2})`, xt, res] ] };
    },
    inter_lagrange: (X, Y, xt) => {
        let n = X.length;
        let sum = 0;
        let debug = [];
        for(let i=0; i<n; i++){
            let L = 1;
            for(let j=0; j<n; j++){
                if(i !== j) L *= (xt - X[j]) / (X[i] - X[j]);
            }
            sum += Y[i] * L;
            debug.push(`L${i}=${L.toFixed(4)}`);
        }
        return { h:["Orde", "Detail L", "Hasil y"], r:[[n-1, debug.join(", "), sum]] };
    },

    // === INTEGRASI ===
    trap_satu: (fs, a, b) => {
        let res = (b-a) * (f(a, fs) + f(b, fs)) / 2;
        return { h:["a","b","Hasil"], r:[[a,b,res]] };
    },
    trap_banyak: (fs, a, b, n) => {
        let h = (b-a)/n;
        let sum = f(a,fs) + f(b,fs);
        for(let i=1; i<n; i++) sum += 2 * f(a+i*h, fs);
        return { h:["h", "n", "Hasil"], r:[[h, n, (h/2)*sum]] };
    },
    simpson_13: (fs, a, b, n) => {
        if(n%2!==0) throw new Error("n harus genap.");
        let h = (b-a)/n;
        let sum = f(a,fs) + f(b,fs);
        for(let i=1; i<n; i++) sum += (i%2===0 ? 2 : 4) * f(a+i*h, fs);
        return { h:["Metode", "Hasil"], r:[["Simpson 1/3", (h/3)*sum]] };
    },
    simpson_38: (fs, a, b, n) => {
        if(n%3!==0) throw new Error("n harus kelipatan 3.");
        let h = (b-a)/n;
        let sum = f(a,fs) + f(b,fs);
        for(let i=1; i<n; i++) sum += (i%3===0 ? 2 : 3) * f(a+i*h, fs);
        return { h:["Metode", "Hasil"], r:[["Simpson 3/8", (3*h/8)*sum]] };
    },
    integral_pias_tak_sama: (X, Y) => {
        // Metode Trapesium untuk data diskrit yang jaraknya tidak sama
        let sum = 0;
        let rows = [];
        for(let i=0; i<X.length-1; i++){
            let h = X[i+1] - X[i];
            let area = (h/2) * (Y[i] + Y[i+1]);
            sum += area;
            rows.push([i+1, X[i], X[i+1], area]);
        }
        return { h:["Segmen", "x1", "x2", "Luas"], r:rows, info: `Total Luas = ${sum}` };
    },
    kuadratur: (fs, a, b, nStr) => {
        // Gauss-Legendre 2 titik atau 3 titik
        let n = parseInt(nStr || "2");
        // Transformasi variabel x = ((b-a)z + (b+a))/2 -> dx = ((b-a)/2) dz
        let c1 = (b-a)/2;
        let c2 = (b+a)/2;
        
        let z = [], w = [];
        if(n === 2) {
            z = [-0.577350269, 0.577350269]; // +/- 1/sqrt(3)
            w = [1, 1];
        } else {
            z = [-0.774596669, 0, 0.774596669]; // -sqrt(0.6), 0, sqrt(0.6)
            w = [0.555555556, 0.888888889, 0.555555556]; // 5/9, 8/9, 5/9
        }
        
        let sum = 0;
        let rows = [];
        for(let i=0; i<z.length; i++){
            let x_real = c1 * z[i] + c2;
            let val = w[i] * f(x_real, fs);
            sum += val;
            rows.push([z[i], w[i], x_real, val]);
        }
        return { h:["z", "w", "x real", "w*f(x)"], r:rows, info: `Hasil Akhir (dikali faktor ${c1}) = ${sum * c1}` };
    },

    // === ODE ===
    euler: (fs, x0, y0, h, xt) => {
        let steps = Math.round((xt-x0)/h);
        let rows = [[0, x0, y0]];
        let y=y0, x=x0;
        for(let i=0; i<steps; i++){
            y += h * fxy(x,y,fs);
            x += h;
            rows.push([i+1, x, y]);
        }
        return { h:["Step","x","y"], r:rows };
    },
    euler_modif: (fs, x0, y0, h, xt) => {
        let steps = Math.round((xt-x0)/h);
        let rows = [[0, x0, y0, "-"]];
        let y=y0, x=x0;
        for(let i=0; i<steps; i++){
            let k1 = fxy(x,y,fs);
            let y_pred = y + h*k1;
            let k2 = fxy(x+h, y_pred, fs);
            y += (h/2)*(k1+k2);
            x += h;
            rows.push([i+1, x, y, (k1+k2)/2]);
        }
        return { h:["Step","x","y","Slope Avg"], r:rows };
    },
    rk4: (fs, x0, y0, h, xt) => {
        let steps = Math.round((xt-x0)/h);
        let rows = [[0, x0, y0]];
        let y=y0, x=x0;
        for(let i=0; i<steps; i++){
            let k1 = fxy(x,y,fs);
            let k2 = fxy(x+0.5*h, y+0.5*h*k1, fs);
            let k3 = fxy(x+0.5*h, y+0.5*h*k2, fs);
            let k4 = fxy(x+h, y+h*k3, fs);
            y += (h/6)*(k1 + 2*k2 + 2*k3 + k4);
            x += h;
            rows.push([i+1, x, y]);
        }
        return { h:["Step","x","y"], r:rows };
    }
};

// =========================================
// 3. UI HANDLER (FORM DINAMIS)
// =========================================
function generateForm() {
    const m = document.getElementById("metode").value;
    const area = document.getElementById("input-area");
    document.getElementById("hasil-container").classList.add("hidden");

    let html = "";
    
    // Kelompok Fungsi f(x)
    if (['bisection','regula','newton','secant','direct_nonlinear','trap_satu','trap_banyak','simpson_13','simpson_38','kuadratur'].includes(m)) {
        html += `<div class="mb-3"><label class="font-bold">Fungsi f(x):</label><input id="func" type="text" class="input input-bordered w-full font-mono" placeholder="x^3 - 2*x - 5"></div>`;
    }

    // Kelompok Input Batas / Tebakan
    if (['bisection','regula','secant'].includes(m)) {
        html += `<div class="grid grid-cols-2 gap-4"><div><label>x1 (Bawah)</label><input id="x1" type="number" step="any" class="input input-bordered w-full"></div><div><label>x2 (Atas)</label><input id="x2" type="number" step="any" class="input input-bordered w-full"></div></div>`;
    }
    if (m === 'direct_nonlinear') {
        html += `<div class="mb-2"><label>Tebakan Awal (x0)</label><input id="x0" type="number" step="any" class="input input-bordered w-full"></div><div class="alert alert-warning text-xs">Penting: Masukkan fungsi dalam bentuk g(x), misal x = sqrt(2x+3).</div>`;
    }
    if (m === 'newton') {
        html += `<div class="mb-2"><label>Turunan f'(x):</label><input id="dfunc" class="input input-bordered w-full font-mono"></div><div><label>x0</label><input id="x0" type="number" step="any" class="input input-bordered w-full"></div>`;
    }

    // Kelompok SPL (Matriks)
    if (['gauss','gauss_jordan','jacobi','gauss_seidel','direct_spl'].includes(m)) {
        html += `<div class="grid grid-cols-2 gap-4">
            <div><label class="font-bold">Matriks A (Koefisien)</label><textarea id="matA" class="textarea textarea-bordered w-full h-32" placeholder="2,1\n3,4"></textarea></div>
            <div><label class="font-bold">Vektor B (Konstanta)</label><textarea id="matB" class="textarea textarea-bordered w-full h-32" placeholder="5\n6"></textarea></div>
        </div>`;
        if (['jacobi','gauss_seidel'].includes(m)) {
            html += `<div class="mt-2"><label>Tebakan Awal (CSV):</label><input id="init" value="0,0,0" class="input input-bordered w-full"></div>`;
        }
    }

    // Kelompok Data Points (Regresi, Interpolasi, Integral Diskrit)
    if (m.startsWith('reg_') || m.startsWith('inter_') || m === 'integral_pias_tak_sama') {
        html += `<div class="mb-2"><label class="font-bold">Data X (CSV):</label><input id="dataX" class="input input-bordered w-full" placeholder="1, 2, 3..."></div>
                 <div class="mb-2"><label class="font-bold">Data Y (CSV):</label><input id="dataY" class="input input-bordered w-full" placeholder="2.5, 3.1, 4.0..."></div>`;
        
        if (m.startsWith('inter_')) {
            html += `<div><label class="font-bold text-primary">Cari nilai pada x = ?</label><input id="xt" type="number" step="any" class="input input-bordered w-full"></div>`;
        }
        if (m === 'reg_polinomial') {
            html += `<div><label class="font-bold">Orde Polinomial (m):</label><input id="orde" type="number" value="2" class="input input-bordered w-full"></div>`;
        }
    }

    // Kelompok Integral Tertentu
    if (['trap_banyak','simpson_13','simpson_38','trap_satu','kuadratur'].includes(m)) {
        html += `<div class="grid grid-cols-2 gap-4 mt-2"><div><label>Batas a</label><input id="a" type="number" step="any" class="input input-bordered w-full"></div><div><label>Batas b</label><input id="b" type="number" step="any" class="input input-bordered w-full"></div></div>`;
        if (!['trap_satu','kuadratur'].includes(m)) {
            html += `<div class="mt-2"><label>Jumlah Pias (n)</label><input id="n" type="number" class="input input-bordered w-full" value="10"></div>`;
        }
        if (m === 'kuadratur') {
             html += `<div class="mt-2"><label>Jumlah Titik (2 atau 3)</label><select id="n" class="select select-bordered w-full"><option value="2">2 Titik</option><option value="3">3 Titik</option></select></div>`;
        }
    }

    // Kelompok ODE
    if (['euler','euler_modif','rk4'].includes(m)) {
        html += `<div class="mb-2"><label class="font-bold">f(x,y) = dy/dx:</label><input id="fode" class="input input-bordered w-full font-mono" value="x+y"></div>
        <div class="grid grid-cols-4 gap-2">
            <div><label>x0</label><input id="x0" type="number" step="any" class="input input-bordered w-full"></div>
            <div><label>y0</label><input id="y0" type="number" step="any" class="input input-bordered w-full"></div>
            <div><label>h</label><input id="h" type="number" step="any" class="input input-bordered w-full"></div>
            <div><label>Target x</label><input id="xt" type="number" step="any" class="input input-bordered w-full"></div>
        </div>`;
    }

    // Iterasi & Toleransi (Umum)
    if (['bisection','regula','newton','secant','direct_nonlinear','jacobi','gauss_seidel'].includes(m)) {
        html += `<div class="grid grid-cols-2 gap-4 mt-4 bg-base-200 p-2 rounded">
            <div><label>Max Iter</label><input id="max" type="number" value="20" class="input input-bordered input-sm w-full"></div>
            <div><label>Toleransi</label><input id="tol" type="number" value="0.0001" step="any" class="input input-bordered input-sm w-full"></div>
        </div>`;
    }

    html += `<button onclick="hitung()" class="btn btn-primary w-full mt-6 text-lg font-bold shadow-lg">HITUNG SEKARANG 🚀</button>`;
    area.innerHTML = html;
}

// =========================================
// 4. MAIN EXECUTOR
// =========================================
function hitung() {
    const m = document.getElementById("metode").value;
    const div = document.getElementById("hasil");
    document.getElementById("hasil-container").classList.remove("hidden");
    div.innerHTML = `<span class="loading loading-dots loading-lg"></span>`;

    try {
        let res;
        // Parsing helpers
        const getNum = (id) => parseFloat(document.getElementById(id).value);
        const getStr = (id) => document.getElementById(id).value;
        const getInt = (id) => parseInt(document.getElementById(id).value);

        // --- NON-LINEAR ---
        if (m === 'bisection') res = Solver.bisection(getStr('func'), getNum('x1'), getNum('x2'), getNum('tol'), getInt('max'));
        else if (m === 'regula') res = Solver.regula(getStr('func'), getNum('x1'), getNum('x2'), getNum('tol'), getInt('max'));
        else if (m === 'direct_nonlinear') res = Solver.direct_nonlinear(getStr('func'), getNum('x0'), getNum('tol'), getInt('max'));
        else if (m === 'newton') res = Solver.newton(getStr('func'), getStr('dfunc'), getNum('x0'), getNum('tol'), getInt('max'));
        else if (m === 'secant') res = Solver.secant(getStr('func'), getNum('x1'), getNum('x2'), getNum('tol'), getInt('max'));

        // --- SPL ---
        else if (m === 'gauss') res = Solver.gauss(parseMatrix('matA'), parseMatrix('matB'));
        else if (m === 'gauss_jordan') res = Solver.gauss_jordan(parseMatrix('matA'), parseMatrix('matB'));
        else if (m === 'direct_spl') res = Solver.direct_spl(parseMatrix('matA'), parseMatrix('matB'));
        else if (m === 'jacobi') res = Solver.jacobi(parseMatrix('matA'), parseMatrix('matB'), parseCSV('init'), getNum('tol'), getInt('max'));
        else if (m === 'gauss_seidel') res = Solver.gauss_seidel(parseMatrix('matA'), parseMatrix('matB'), parseCSV('init'), getNum('tol'), getInt('max'));

        // --- REGRESI ---
        else if (m === 'reg_linier') res = Solver.reg_linier(parseCSV('dataX'), parseCSV('dataY'));
        else if (m === 'reg_nonlinier') res = Solver.reg_nonlinier(parseCSV('dataX'), parseCSV('dataY'));
        else if (m === 'reg_polinomial') res = Solver.reg_polinomial(parseCSV('dataX'), parseCSV('dataY'), getNum('orde'));

        // --- INTERPOLASI ---
        else if (m === 'inter_linier') res = Solver.inter_linier(parseCSV('dataX'), parseCSV('dataY'), getNum('xt'));
        else if (m === 'inter_kuadrat') res = Solver.inter_kuadrat(parseCSV('dataX'), parseCSV('dataY'), getNum('xt'));
        else if (m === 'inter_lagrange') res = Solver.inter_lagrange(parseCSV('dataX'), parseCSV('dataY'), getNum('xt'));

        // --- INTEGRASI ---
        else if (m === 'trap_satu') res = Solver.trap_satu(getStr('func'), getNum('a'), getNum('b'));
        else if (m === 'trap_banyak') res = Solver.trap_banyak(getStr('func'), getNum('a'), getNum('b'), getInt('n'));
        else if (m === 'simpson_13') res = Solver.simpson_13(getStr('func'), getNum('a'), getNum('b'), getInt('n'));
        else if (m === 'simpson_38') res = Solver.simpson_38(getStr('func'), getNum('a'), getNum('b'), getInt('n'));
        else if (m === 'integral_pias_tak_sama') res = Solver.integral_pias_tak_sama(parseCSV('dataX'), parseCSV('dataY'));
        else if (m === 'kuadratur') res = Solver.kuadratur(getStr('func'), getNum('a'), getNum('b'), getStr('n'));

        // --- ODE ---
        else if (['euler','euler_modif','rk4'].includes(m)) {
            let func = getStr('fode'), x0=getNum('x0'), y0=getNum('y0'), h=getNum('h'), xt=getNum('xt');
            if(m === 'euler') res = Solver.euler(func, x0, y0, h, xt);
            else if(m === 'euler_modif') res = Solver.euler_modif(func, x0, y0, h, xt);
            else res = Solver.rk4(func, x0, y0, h, xt);
        }

        // Output Result
        if(res) div.innerHTML = createTable(res.h, res.r, res.info);
        else throw new Error("Metode belum diimplementasi sepenuhnya.");

    } catch (err) {
        console.error(err);
        div.innerHTML = `<div class="alert alert-error shadow-lg">
            <svg xmlns="http://www.w3.org/2000/svg" class="stroke-current flex-shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
            <span>Error: ${err.message}</span>
        </div>`;
    }
}

// Init Event Listeners
document.getElementById("metode").addEventListener("change", generateForm);
document.addEventListener("DOMContentLoaded", generateForm);