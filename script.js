lucide.createIcons();

// --- STATE ---
let map = L.map('map').setView([-2.5, 118], 5);
let quakeLayer = L.layerGroup().addTo(map);
let weatherLayer = L.layerGroup().addTo(map);
let userMarker = null;
let model = null;
let trendChart = null;
let allQuakesData = []; // Global store for analysis
let allWeatherData = {}; // Global store for weather
let userCoords = null;
const apiKey = "";

// --- GEOLOGICAL KNOWLEDGE BASE ---
const GEO_FACTS = [
    "Lempeng tektonik bergerak secepat pertumbuhan kuku manusia (sekitar 2-5 cm per tahun).",
    "Indonesia berada di 'Ring of Fire', jalur bencana yang membentang 40.000 km di sepanjang Samudra Pasifik.",
    "Gempa bumi terdalam yang pernah tercatat terjadi pada kedalaman lebih dari 700 km di bawah permukaan bumi.",
    "Tsunami dapat melintasi samudra dengan kecepatan yang sama dengan pesawat jet (hingga 800 km/jam).",
    "Bumi memiliki sekitar 12 lempeng tektonik utama yang terus bergeser dan bertabrakan.",
    "Gempa bumi berkekuatan 9.0 melepaskan sekitar 1.000 kali lebih banyak energi daripada gempa berkekuatan 7.0."
];
const GEO_TOPICS = ["Plate_tectonics", "Seismology", "Richter_magnitude_scale", "Ring_of_Fire", "Tsunami", "Sunda_Megathrust"];

// Haversine Distance Helper
function calculateDistance(lat1, lon1, lat2, lon2) {
    const R = 6371; // km
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
        Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
}

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© NusantaraAI'
}).addTo(map);

// --- AI ENGINE (Neural Risk) ---
async function initAI() {
    try {
        model = tf.sequential();
        model.add(tf.layers.dense({ units: 8, inputShape: [2], activation: 'relu' }));
        model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
        await model.compile({ optimizer: 'sgd', loss: 'categoricalCrossentropy' });
        const xs = tf.tensor2d([[1, 100], [5, 50], [8, 10], [4, 200]]);
        const ys = tf.tensor2d([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]);
        await model.fit(xs, ys, { epochs: 5 });
        renderSuggestions();
    } catch (e) {
        console.error("Neural Init Failed", e);
    }
}

async function analyzeQuake(mag, depthStr) {
    if (!model) return { level: 'MODERAT', score: 50, confidence: 'BASE' };
    const depth = parseInt(depthStr.replace(/\D/g, '')) || 10;
    const prediction = tf.tidy(() => {
        const input = tf.tensor2d([[parseFloat(mag), depth]]);
        return model.predict(input).dataSync();
    });
    const labels = ["RENDAH", "MODERAT", "TINGGI"];
    const maxIdx = prediction.indexOf(Math.max(...prediction));
    return {
        level: labels[maxIdx],
        score: (prediction[maxIdx] * 100).toFixed(1),
        confidence: prediction[maxIdx] > 0.7 ? 'HIGH_CONFIDENCE' : 'BASE_ESTIMATION'
    };
}

// --- SEISMIC FORECASTING (TF.js Time-Series) ---
async function runForecasting() {
    const trendEl = document.getElementById('forecast-trend');
    const magEl = document.getElementById('forecast-mag');
    const descEl = document.getElementById('forecast-desc');

    if (!trendEl || !magEl || !descEl || allQuakesData.length < 5) {
        if (trendEl) trendEl.innerText = "Data Terbatas";
        return;
    }

    try {
        // Collect last 15 magnitudes (local + global)
        const data = allQuakesData.slice(0, 15).map(q => parseFloat(q.Magnitude)).reverse();
        const xs = tf.tensor2d(data.map((_, i) => [i]), [data.length, 1]);
        const ys = tf.tensor2d(data, [data.length, 1]);

        // Simple Linear Regression Model
        const forecastModel = tf.sequential();
        forecastModel.add(tf.layers.dense({ units: 1, inputShape: [1] }));
        forecastModel.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

        await forecastModel.fit(xs, ys, { epochs: 20, verbose: 0 });

        // Predict next step
        const nextIdx = data.length;
        const prediction = forecastModel.predict(tf.tensor2d([[nextIdx]], [1, 1])).dataSync()[0];
        const lastMag = data[data.length - 1];

        // UI Handling
        const trend = prediction > lastMag ? 'MENINGKAT' : 'MENURUN';
        trendEl.innerText = `TREN BIAS: ${trend}`;
        trendEl.className = `text-xs font-bold ${trend === 'MENINGKAT' ? 'text-danger' : 'text-emerald-400'} uppercase tracking-widest italic animate-pulse`;
        magEl.innerText = `${Math.abs(prediction).toFixed(1)} M`;

        descEl.innerText = trend === 'MENINGKAT'
            ? "Neural pattern mendeteksi bias peningkatan intensitas. Tetap siaga di wilayah rawan."
            : "Data menunjukkan fase relaksasi seismik sementara. Tetap waspada terhadap aftershocks.";

        // Cleanup
        xs.dispose();
        ys.dispose();
        forecastModel.dispose();
    } catch (e) {
        console.error("Forecasting Error", e);
    }
}

// --- WEATHER & LOCATION ---
async function getWeatherData(lat, lon) {
    try {
        const res = await axios.get(`https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current_weather=true`);
        const weather = res.data.current_weather;
        const codes = {
            0: "Cerah", 1: "Cerah Berawan", 2: "Berawan", 3: "Mendung",
            45: "Kabut", 48: "Kabut", 51: "Gerimis", 61: "Hujan Ringan",
            80: "Hujan", 95: "Badai Petir"
        };
        document.getElementById('stat-weather-val').innerText = codes[weather.weathercode] || "Normal";
        document.getElementById('stat-temp').innerText = `${weather.temperature}°C`;
    } catch (e) {
        console.error("Weather failed", e);
    }
}

const REGIONAL_CITIES = [
    { name: "Jakarta", lat: -6.2088, lon: 106.8456 },
    { name: "Surabaya", lat: -7.2575, lon: 112.7521 },
    { name: "Medan", lat: 3.5952, lon: 98.6722 },
    { name: "Makassar", lat: -5.1476, lon: 119.4327 },
    { name: "Jayapura", lat: -2.5337, lon: 140.7181 },
    { name: "Denpasar", lat: -8.6705, lon: 115.2126 },
    { name: "Banjarmasin", lat: -3.3167, lon: 114.5901 },
    { name: "Palembang", lat: -2.9761, lon: 104.7754 },
    { name: "Bandung", lat: -6.9175, lon: 107.6191 },
    { name: "Semarang", lat: -6.9667, lon: 110.4167 },
    { name: "Yogyakarta", lat: -7.7956, lon: 110.3695 },
    { name: "Pontianak", lat: -0.0263, lon: 109.3425 },
    { name: "Samarinda", lat: -0.5022, lon: 117.1536 },
    { name: "Manado", lat: 1.4748, lon: 124.8420 },
    { name: "Ambon", lat: -3.6547, lon: 128.1906 },
    { name: "Kupang", lat: -10.1772, lon: 123.6070 },
    { name: "Mataram", lat: -8.5799, lon: 116.0891 },
    { name: "Padang", lat: -0.9471, lon: 100.4172 },
    { name: "Pekanbaru", lat: 0.5071, lon: 101.4478 },
    { name: "Banda Aceh", lat: 5.5483, lon: 95.3238 },
    { name: "Balikpapan", lat: -1.2654, lon: 116.8312 },
    { name: "Lampung", lat: -5.4292, lon: 105.2611 },
    { name: "Sorong", lat: -0.8762, lon: 131.2558 },
    { name: "Ternate", lat: 0.7905, lon: 127.3824 },
    { name: "Gorontalo", lat: 0.5435, lon: 123.0568 },
    { name: "Palu", lat: -0.8917, lon: 119.8707 }
];

async function updateRegionalWeather() {
    weatherLayer.clearLayers();
    const weatherIcons = {
        0: "sun", 1: "cloud-sun", 2: "cloud", 3: "cloud", 45: "haze", 48: "haze",
        51: "cloud-drizzle", 61: "cloud-rain", 80: "cloud-lightning", 95: "cloud-lightning"
    };
    const codes = {
        0: "Cerah", 1: "Cerah Berawan", 2: "Berawan", 3: "Mendung",
        45: "Kabut", 48: "Kabut", 51: "Gerimis", 61: "Hujan Ringan",
        80: "Hujan", 95: "Badai Petir"
    };

    const weatherPromises = REGIONAL_CITIES.map(async (city) => {
        try {
            const res = await axios.get(`https://api.open-meteo.com/v1/forecast?latitude=${city.lat}&longitude=${city.lon}&current_weather=true`);
            const data = res.data.current_weather;
            const iconName = weatherIcons[data.weathercode] || "cloud";

            allWeatherData[city.name.toLowerCase()] = {
                ...data,
                condition: codes[data.weathercode] || "Normal",
                cityName: city.name
            };

            const weatherDiv = L.divIcon({
                className: 'weather-icon-marker',
                html: `<div class="flex flex-col items-center bg-white/10 backdrop-blur-md p-1.5 rounded-xl border border-white/20 shadow-lg group hover:bg-white/20 transition-all cursor-pointer">
                        <i data-lucide="${iconName}" class="w-3.5 h-3.5 text-warning"></i>
                        <span class="text-[7px] font-bold text-white mt-1 hidden group-hover:block">${data.temperature}°C</span>
                       </div>`,
                iconSize: [30, 30],
                iconAnchor: [15, 15]
            });

            L.marker([city.lat, city.lon], { icon: weatherDiv })
                .addTo(weatherLayer)
                .bindPopup(`<div class="text-xs"><b>${city.name}</b><br>Suhu: ${data.temperature}°C<br>Kondisi: ${allWeatherData[city.name.toLowerCase()].condition}</div>`);
        } catch (e) {
            console.error(`Regional weather failed for ${city.name}`, e);
        }
    });

    await Promise.all(weatherPromises);
    lucide.createIcons();
}

function getUserLocation() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(pos => {
            const { latitude, longitude } = pos.coords;
            document.getElementById('user-location-text').innerText = `${latitude.toFixed(2)}, ${longitude.toFixed(2)}`;
            userCoords = { lat: latitude, lon: longitude };

            if (userMarker) map.removeLayer(userMarker);
            userMarker = L.circleMarker([latitude, longitude], {
                radius: 8,
                fillColor: '#a855f7',
                color: '#fff',
                weight: 2,
                fillOpacity: 1
            }).addTo(map).bindPopup("Lokasi Anda").openPopup();

            getWeatherData(latitude, longitude);
            updateLocalNowcast(); // Immediate update on location lock
        }, () => {
            document.getElementById('user-location-text').innerText = "Akses ditolak";
        });
    }
}

// --- CHARTING ---
function updateTrendChart(quakes) {
    const ctx = document.getElementById('quakeTrendChart').getContext('2d');

    // Split data by source and take recent samples
    const bmkgData = allQuakesData.filter(q => q.source === 'BMKG').slice(0, 10).reverse();
    const usgsData = allQuakesData.filter(q => q.source === 'USGS').slice(0, 10).reverse();

    // Identify labels (use index or time if consistent)
    const labels = Array.from({ length: 10 }, (_, i) => i + 1);

    if (trendChart) trendChart.destroy();

    trendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'BMKG (Local)',
                    data: bmkgData.map(q => parseFloat(q.Magnitude)),
                    borderColor: '#0ea5e9',
                    backgroundColor: 'rgba(14, 165, 233, 0.1)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 3
                },
                {
                    label: 'USGS (Global)',
                    data: usgsData.map(q => parseFloat(q.Magnitude)),
                    borderColor: '#a855f7',
                    backgroundColor: 'rgba(168, 85, 247, 0.1)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: { color: '#64748b', font: { size: 10 } }
                }
            },
            scales: {
                y: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#64748b', font: { size: 10 } }
                },
                x: {
                    display: false,
                    grid: { display: false }
                }
            }
        }
    });
}

// --- CHATBOT BUBBLE LOGIC ---
window.toggleChat = function () {
    const win = document.getElementById('chat-window');
    win.classList.toggle('chat-bubble-active');
};

window.toggleExpandChat = function () {
    const win = document.getElementById('chat-window');
    const icon = document.getElementById('chat-resize-icon');
    win.classList.toggle('chat-expanded');

    if (win.classList.contains('chat-expanded')) {
        icon.setAttribute('data-lucide', 'minimize-2');
    } else {
        icon.setAttribute('data-lucide', 'maximize-2');
    }
    lucide.createIcons();
};

function renderSuggestions() {
    const list = ["Pertolongan Pertama", "Rute Evakuasi", "Info SiagaBot", "Tren Gempa"];
    const container = document.getElementById('chat-suggestions');
    if (!container) return;
    container.innerHTML = list.map(s => `
        <button onclick="handleChat('${s}')" class="whitespace-nowrap px-3 py-1 bg-white/5 border border-white/10 rounded-full text-[9px] font-bold text-slate-400 hover:text-white hover:bg-white/10 transition-all uppercase tracking-tighter">
            ${s}
        </button>
    `).join('');
}

async function handleChat(preMsg) {
    const input = document.getElementById('chat-input');
    const box = document.getElementById('chat-box');
    const msgText = preMsg || input.value;

    if (!msgText.trim()) return;

    // User Message
    const userDiv = document.createElement('div');
    userDiv.className = "bg-neural/10 p-3 rounded-2xl border border-neural/20 ml-8 text-right self-end";
    userDiv.innerText = msgText;
    box.appendChild(userDiv);

    let response = "Maaf, **SiagaBot** sedang memproses data. Coba tanyakan tentang 'pertolongan pertama', 'rute evakuasi', atau 'grafik'.";
    let chartId = null;
    let extraHTML = "";
    const q = msgText.toLowerCase();

    if (q.includes("hi") || q.includes("halo") || q.includes("siagabot")) {
        response = `Halo! Saya **SiagaBot**, asisten keselamatan Anda. Saya dapat membantu memberikan panduan pertolongan pertama, rute evakuasi, atau analisis gempa terbaru secara real-time. Tanyakan sesuatu seperti "Apa yang harus dilakukan saat gempa?"`;
    } else if (q.includes("p3k") || q.includes("pertolongan pertama") || q.includes("luka")) {
        response = `### 🚑 Panduan Pertolongan Pertama (Earthquake First Aid)
Berdasarkan protokol medis darurat:
1. **Lakukan Penilaian Cepat**: Cek kesadaran dan pernapasan korban.
2. **Pendarahan Hebat**: Tekan sumber pendarahan dengan kain bersih. Gunakan tourniquet jika pendarahan di tangan/kaki tidak berhenti.
3. **Patah Tulang**: Immobilisasi bagian yang patah. Jangan mencoba mengembalikan posisi tulang.
4. **Resusitasi (CPR)**: Jika tidak ada denyut/napas, lakukan kompresi dada 100-120x per menit (hanya jika Anda terlatih).
5. **Shock**: Baringkan korban, tinggikan kaki 30cm, dan jaga suhu tubuh agar tetap hangat.`;
    } else if (q.includes("dalam") || q.includes("indoor") || q.includes("gedung")) {
        response = `### 🏢 Panduan Saat di Dalam Ruangan (Drop, Cover, Hold On)
Langkah resmi dari **BMKG & USGS**:
1. **DROP! (Merunduk)**: Segera merunduk ke lantai sebelum guncangan menjatuhkan Anda.
2. **COVER! (Berlindung)**: Berlindung di bawah meja yang kokoh. Jika tidak ada meja, lindungi kepala dan leher dengan lengan Anda.
3. **HOLD ON! (Bertahan)**: Pegang kaki meja tersebut dan tetap di sana sampai guncangan berhenti.
4. **Hindari**: Lift, tangga darurat (saat guncangan), jendela kaca, dan lemari berat.`;
    } else if (q.includes("luar") || q.includes("outdoor") || q.includes("lapangan")) {
        response = `### 🌳 Panduan Saat di Luar Ruangan
Prioritas utama adalah area terbuka:
1. **Cari Area Terbuka**: Menjauhlah dari gedung, dinding, papan reklame, dan tiang listrik.
2. **Hati-hati Bahaya Atas**: Waspadai jatuhnya kaca jendela atau ornamen gedung (seringkali area paling berbahaya adalah dekat dinding luar gedung).
3. **Tetap di Sana**: Jangan mencoba lari ke dalam gedung. Tunggu sampai guncangan benar-benar berhenti.`;
    } else if (q.includes("tsunami") || q.includes("laut") || q.includes("pantai")) {
        response = `### 🌊 Mitigasi Madiri Tsunami (Aturan 20-20-20)
Jika Anda berada di dekat pantai:
1. **20 Detik**: Jika guncangan gempa terasa lebih lama dari **20 detik**.
2. **20 Menit**: Anda hanya punya waktu sekitar **20 menit** untuk evakuasi sebelum gelombang pertama tiba.
3. **20 Meter**: Segera lari ke tempat yang tingginya minimal **20 meter** di atas permukaan laut.
**JANGAN TUNGGU SIRINE!** Gempa besar adalah peringatan tsunami alami yang paling utama.`;
    } else if (q.includes("rute") || q.includes("evakuasi") || q.includes("jalur")) {
        response = `### Tips Rute Evakuasi & Penyelamatan
1. **Identifikasi Jalur Keluar**: Gunakan tangga darurat, **JANGAN GUNAKAN LIFT**.
2. **Titik Kumpul**: Menuju ke area terbuka (lapangan/parkir) yang jauh dari gedung dan tiang listrik.
3. **Navigasi Offline**: Hafalkan rute utama ke tempat tinggi jika ada ancaman tsunami. 
4. **Tas Siaga**: Pastikan Anda membawa masker, senter, dan air minum saat proses evakuasi.`;
    } else if (q.includes("risiko") || q.includes("analisis")) {
        const mag = document.getElementById('stat-latest').innerText;
        response = `Berdasarkan analisis Neural terbaru, aktivitas magnitudo terakhir tercatat sebesar **${mag}**. Risiko wilayah dianalisis secara real-time berdasarkan kedalaman sesar dan pola historis. Silakan buka laporan AI pada marker merah berdenyut untuk detail lebih lanjut.`;
    } else if (q.includes("terdekat") || q.includes("dekat") || q.includes("radius")) {
        if (!userCoords) {
            response = "Saya butuh akses lokasi (GPS) Anda untuk mencari gempa terdekat. Mohon izinkan akses lokasi pada browser Anda.";
        } else if (allQuakesData.length === 0) {
            response = "Data gempa belum tersedia atau sedang dimuat. Mohon tunggu sebentar.";
        } else {
            const sorted = [...allQuakesData].map(quake => {
                const [qLat, qLon] = quake.Coordinates.split(',').map(parseFloat);
                return { ...quake, dist: calculateDistance(userCoords.lat, userCoords.lon, qLat, qLon) };
            }).sort((a, b) => a.dist - b.dist);

            const nearest = sorted[0];
            const distStr = nearest.dist < 1 ? `${(nearest.dist * 1000).toFixed(0)} meter` : `${nearest.dist.toFixed(1)} km`;
            response = `Menemukan kejadian terdekat: Gempa di **${nearest.Wilayah}** (${nearest.Magnitude} M). Berjarak sekitar **${distStr}** dari posisi Anda sekarang. Tetap waspada!`;
        }
    } else if (q.includes("grafik") || q.includes("tren") || q.includes("history")) {
        chartId = `chat-chart-${Date.now()}`;
        response = "Ini adalah visualisasi tren magnitudo dari 10 aktivitas seismik terbaru di Nusantara:";
        extraHTML = `<div class="mt-3 h-32 w-full"><canvas id="${chartId}"></canvas></div>`;
    } else if (q.includes("prediksi") || q.includes("ramalan") || q.includes("forecast")) {
        const trend = document.getElementById('forecast-trend').innerText;
        const mag = document.getElementById('forecast-mag').innerText;
        response = `Berdasarkan analisis bias data terbaru menggunakan **Seismic Forecast Model (TensorFlow.js)**, potensi aktivitas magnitude berikutnya diperkirakan sekitar **${mag}**. Status tren saat ini menunjukkan kondisi **${trend}**. Harap diingat bahwa ini adalah prediksi matematis berbasis tren, tetap ikuti instruksi resmi dari BMKG.`;
    } else if (q.includes("bandingkan") || q.includes("perbandingan") || q.includes("terbesar")) {
        if (allQuakesData.length > 0) {
            const top5 = [...allQuakesData].sort((a, b) => parseFloat(b.Magnitude) - parseFloat(a.Magnitude)).slice(0, 5);
            response = "Berikut adalah perbandingan 5 kejadian gempa dengan magnitudo terbesar dari data saat ini:";
            extraHTML = `<div class="mt-3 overflow-x-auto border border-white/10 rounded-xl">
                <table class="w-full text-[9px] text-left">
                    <tr class="bg-white/10 text-slate-400 uppercase font-bold"><th class="p-2">Lokasi</th><th class="p-2 text-center">Mag</th><th class="p-2">Waktu</th></tr>
                    ${top5.map(t => `<tr class="border-t border-white/5"><td class="p-2 truncate max-w-[100px] text-white font-medium">${t.Wilayah}</td><td class="p-2 text-center font-bold text-neural">${t.Magnitude}</td><td class="p-2 text-slate-500">${t.Jam.split(' ')[0]}</td></tr>`).join('')}
                </table>
            </div>`;
        } else {
            response = "Data tidak cukup untuk melakukan perbandingan saat ini.";
        }
    } else if (q.includes("gempa") || q.includes("status")) {
        const count = allQuakesData.length;
        response = `Saat ini saya memantau **${count}** aktivitas seismik dari database BMKG. Kejadian terbaru berada di wilayah **${allQuakesData[0]?.Wilayah || 'N/A'}**.`;
    } else if (q.includes("tips") || q.includes("aman") || q.includes("siaga")) {
        response = "Siaga 24/7! Selalu ingat: 1. **Tiarap** (Drop), 2. **Lindungi** kepala (Cover), 3. **Bertahan** (Hold on). Jauhi kaca, pohon, dan tiang listrik.";
    } else if (q.includes("cuaca")) {
        const cityMatch = REGIONAL_CITIES.find(c => q.includes(c.name.toLowerCase()));

        if (cityMatch && allWeatherData[cityMatch.name.toLowerCase()]) {
            const w = allWeatherData[cityMatch.name.toLowerCase()];
            response = `Kondisi cuaca di **${w.cityName}** saat ini: **${w.condition}** dengan suhu **${w.temperature}°C**.`;
        } else if (q.includes("seluruh") || q.includes("indonesia") || q.includes("nasional")) {
            const cities = Object.values(allWeatherData);
            if (cities.length > 0) {
                const avgTemp = (cities.reduce((sum, c) => sum + c.temperature, 0) / cities.length).toFixed(1);
                response = `Rangkuman cuaca nasional: Rata-rata suhu di **${cities.length}** kota besar adalah **${avgTemp}°C**. Sebagian besar wilayah terpantau **${cities[0].condition}**. Ketik nama kota untuk detail spesifik.`;
            } else {
                response = "Data cuaca nasional sedang dimuat. Harap tunggu beberapa saat.";
            }
        } else {
            const userW = document.getElementById('stat-weather-val').innerText;
            const userT = document.getElementById('stat-temp').innerText;
            response = `Info cuaca di lokasi Anda: **${userW}** dengan suhu **${userT}**. Untuk kota lain, silakan ketik (misal: "cuaca Jakarta").`;
        }
    } else if (q.includes("halo") || q.includes("hi") || q.includes("siapa")) {
        response = "Halo! Saya adalah **GEORIS Assistant**, asisten cerdas yang memantau keamanan geo-spasial Indonesia. Ada yang bisa saya bantu?";
    }

    // AI Response (Rendered)
    const botDiv = document.createElement('div');
    botDiv.className = "bg-white/5 p-3 rounded-2xl border border-white/5 mr-8 flex flex-col gap-2";

    // Ensure marked is available and use it correctly
    let parsedMarkdown = response;
    try {
        if (typeof marked !== 'undefined') {
            parsedMarkdown = marked.parse(response);
        }
    } catch (e) {
        console.error("Markdown parse failed", e);
    }

    botDiv.innerHTML = `
        <div class="flex gap-2">
            <i data-lucide="bot" class="w-3 h-3 text-neural shrink-0"></i>
            <div class="leading-relaxed markdown-content">${parsedMarkdown}</div>
        </div>
        ${extraHTML}
    `;

    setTimeout(() => {
        box.appendChild(botDiv);
        box.scrollTop = box.scrollHeight;
        lucide.createIcons();

        if (chartId && allQuakesData.length > 0) {
            const ctx = document.getElementById(chartId).getContext('2d');
            const bmkgSlice = allQuakesData.filter(d => d.source === 'BMKG').slice(0, 8).reverse();
            const usgsSlice = allQuakesData.filter(d => d.source === 'USGS').slice(0, 8).reverse();

            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Array.from({ length: 8 }, (_, i) => i + 1),
                    datasets: [
                        {
                            label: 'Local (BMKG)',
                            data: bmkgSlice.map(d => parseFloat(d.Magnitude)),
                            borderColor: '#0ea5e9',
                            backgroundColor: 'rgba(14, 165, 233, 0.2)',
                            fill: true,
                            tension: 0.4,
                            borderWidth: 2,
                            pointRadius: 2
                        },
                        {
                            label: 'Global (USGS)',
                            data: usgsSlice.map(d => parseFloat(d.Magnitude)),
                            borderColor: '#a855f7',
                            backgroundColor: 'rgba(168, 85, 247, 0.2)',
                            fill: true,
                            tension: 0.4,
                            borderWidth: 2,
                            pointRadius: 2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            labels: { color: '#64748b', font: { size: 8 } }
                        }
                    },
                    scales: {
                        y: {
                            display: true,
                            suggestedMin: 2,
                            suggestedMax: 8,
                            grid: { color: 'rgba(255,255,255,0.05)' },
                            ticks: { color: '#64748b', font: { size: 6 } }
                        },
                        x: {
                            display: false
                        }
                    }
                }
            });
        }
    }, 600);

    input.value = "";
    box.scrollTop = box.scrollHeight;
}

// --- MODAL & FEED (DENGAN DATA LENGKAP) ---
async function openAIModal(q) {
    const modal = document.getElementById('ai-modal');
    const body = document.getElementById('modal-body');
    modal.classList.add('modal-active');
    body.innerHTML = `<div class="flex flex-col items-center justify-center h-64 opacity-50"><div class="animate-spin mb-4"><i data-lucide="loader-2" class="w-8 h-8 text-neural"></i></div><span class="text-[10px] uppercase font-bold tracking-widest">Neural Processing CAP Data...</span></div>`;
    lucide.createIcons();

    const analysis = await analyzeQuake(q.Magnitude, q.Kedalaman);
    const riskColor = analysis.level === 'TINGGI' ? 'text-danger' : (analysis.level === 'MODERAT' ? 'text-warning' : 'text-emerald-400');
    const riskBorder = analysis.level === 'TINGGI' ? 'border-danger/30' : (analysis.level === 'MODERAT' ? 'border-warning/30' : 'border-emerald-500/30');

    // Metadata Nowcast (Mocked or extrapolated from BMKG feed attributes)
    const capData = {
        event: q.Potensi || "Aktivitas Seismik Signifikan",
        effective: q.Jam || "Waktu Kejadian",
        expires: "N/A (Cek update berkala)",
        senderName: "BMKG Indonesia (TEWS)",
        description: q.Wilayah || "Wilayah terdampak sesuai koordinat sensor.",
        area: q.Coordinates || "Tersedia di peta"
    };

    body.innerHTML = `
        <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
            <!-- Data BMKG & CAP -->
            <div class="space-y-6">
                <div class="flex items-center gap-2">
                    <i data-lucide="database" class="w-4 h-4 text-accent"></i>
                    <h4 class="text-[10px] font-bold text-white uppercase tracking-widest">Detail Parameter (CAP)</h4>
                </div>
                <div class="space-y-4">
                    <div class="p-4 bg-white/5 rounded-2xl border border-white/5">
                        <span class="text-[9px] text-slate-500 uppercase block mb-1">Jenis Kejadian (Event)</span>
                        <span class="text-xs font-bold text-white">${capData.event}</span>
                    </div>
                    <div class="p-4 bg-white/5 rounded-2xl border border-white/5">
                        <span class="text-[9px] text-slate-500 uppercase block mb-1">Wilayah Terdampak (Description)</span>
                        <span class="text-xs text-slate-300 leading-relaxed">${capData.description}</span>
                    </div>
                    <div class="grid grid-cols-2 gap-3">
                        <div class="p-3 bg-white/5 rounded-2xl border border-white/5">
                            <span class="text-[9px] text-slate-500 uppercase block mb-1">Mulai (Effective)</span>
                            <span class="text-[10px] font-bold text-white">${capData.effective}</span>
                        </div>
                        <div class="p-3 bg-white/5 rounded-2xl border border-white/5">
                            <span class="text-[9px] text-slate-500 uppercase block mb-1">Selesai (Expires)</span>
                            <span class="text-[10px] font-bold text-slate-500">${capData.expires}</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Neural Analysis -->
            <div class="space-y-6">
                <div class="flex items-center gap-2">
                    <i data-lucide="brain-circuit" class="w-4 h-4 text-neural"></i>
                    <h4 class="text-[10px] font-bold text-white uppercase tracking-widest">Neural Assessment</h4>
                </div>
                <div class="p-5 bg-neural/5 rounded-2xl border-2 ${riskBorder} relative overflow-hidden">
                    <div class="scan-line"></div>
                    <div class="flex justify-between items-start mb-4">
                        <div>
                            <span class="text-[9px] text-slate-500 uppercase block mb-1">Risk Level</span>
                            <span class="text-2xl font-display font-black ${riskColor}">${analysis.level}</span>
                        </div>
                        <div class="text-right">
                            <span class="text-[9px] text-slate-500 uppercase block mb-1">Impact Score</span>
                            <span class="text-xl font-display font-bold text-white">${analysis.score}%</span>
                        </div>
                    </div>
                    <div class="h-1 w-full bg-white/10 rounded-full">
                        <div class="h-full bg-neural transition-all duration-1000" style="width: ${analysis.score}%"></div>
                    </div>
                    <div class="mt-6 pt-6 border-t border-white/5">
                        <p class="text-[10px] text-slate-400 italic">
                            <b>Rekomendasi AI:</b> ${analysis.level === 'TINGGI' ? 'Segera cari tempat terbuka. Jauhi gedung bertingkat dan kabel listrik.' : 'Tetap tenang dan pantau informasi resmi BMKG untuk update Nowcast.'}
                        </p>
                    </div>
                </div>
                <div class="p-4 bg-white/5 rounded-2xl border border-white/5">
                    <span class="text-[9px] text-slate-500 uppercase block mb-1">Instansi Pengirim</span>
                    <span class="text-xs font-bold text-white">${capData.senderName}</span>
                </div>
            </div>
        </div>
    `;
    lucide.createIcons();
}

window.closeModal = function () {
    document.getElementById('ai-modal').classList.remove('modal-active');
};

async function fetchData() {
    try {
        const [auto, recent, global] = await Promise.all([
            axios.get("https://data.bmkg.go.id/DataMKG/TEWS/autogempa.json"),
            axios.get("https://data.bmkg.go.id/DataMKG/TEWS/gempaterkini.json"),
            axios.get("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson")
        ]);

        const autoQuake = auto.data?.Infogempa?.gempa;
        const recentQuakes = (recent.data?.Infogempa?.gempa || []).map(q => ({ ...q, source: 'BMKG' }));
        const globalQuakes = (global.data?.features || []).map(f => ({
            Jam: new Date(f.properties.time).toLocaleString('id-ID'),
            Coordinates: `${f.geometry.coordinates[1]}, ${f.geometry.coordinates[0]}`,
            Magnitude: f.properties.mag.toString(),
            Kedalaman: `${f.geometry.coordinates[2]} km`,
            Wilayah: f.properties.place,
            Potensi: "USGS Global Data",
            Dirasakan: "-",
            source: 'USGS'
        }));

        // Enhanced deduplication for BMKG
        let mergedLocal = [...recentQuakes];
        if (autoQuake) {
            const isDuplicate = mergedLocal.some(q =>
                q.Jam.trim() === autoQuake.Jam.trim() &&
                q.Coordinates.replace(/\s/g, '') === autoQuake.Coordinates.replace(/\s/g, '')
            );
            if (!isDuplicate) mergedLocal.unshift({ ...autoQuake, source: 'BMKG' });
        }

        allQuakesData = [...mergedLocal, ...globalQuakes];
        renderQuakes(allQuakesData, autoQuake);
        renderFeed(allQuakesData); // Pass all data to feed
        updateTrendChart(mergedLocal);
        updateLocalNowcast();

        if (mergedLocal[0]) {
            document.getElementById('stat-latest').innerText = `${mergedLocal[0].Magnitude} M`;
            document.getElementById('stat-loc').innerText = mergedLocal[0].Wilayah;
        }

        // Run Seismic Forecasting
        runForecasting();
    } catch (e) {
        console.error("Data fetch failed", e);
    }
}

function renderQuakes(quakes, latest) {
    quakeLayer.clearLayers();
    quakes.forEach(q => {
        if (!q.Coordinates) return;
        const c = q.Coordinates.split(',');
        const lat = parseFloat(c[0]);
        const lon = parseFloat(c[1]);
        const mag = parseFloat(q.Magnitude);
        const source = q.source || 'BMKG';

        let color = mag >= 6 ? '#ef4444' : (mag >= 5 ? '#f59e0b' : '#0ea5e9');
        if (source === 'USGS') {
            color = '#a855f7'; // Purple for global
        }

        const popupContent = `
            <div class="p-2 space-y-2 min-w-[150px]">
                <div class="flex items-center gap-2 border-b border-slate-700 pb-2 mb-2">
                    <div class="w-2 h-2 rounded-full ${source === 'USGS' ? 'bg-neural' : 'bg-accent'} animate-ping"></div>
                    <span class="text-[10px] font-bold text-slate-400 uppercase tracking-widest">${source} Earthquake</span>
                </div>
                <div class="text-xs font-bold text-white mb-1">${q.Wilayah}</div>
                <div class="grid grid-cols-2 gap-2 text-[10px]">
                    <div class="bg-white/5 p-1 rounded-md text-center">
                        <span class="block text-slate-500 uppercase">Mag</span>
                        <span class="font-bold text-accent">${q.Magnitude}</span>
                    </div>
                    <div class="bg-white/5 p-1 rounded-md text-center">
                        <span class="block text-slate-500 uppercase">Kedalaman</span>
                        <span class="font-bold text-white">${q.Kedalaman}</span>
                    </div>
                </div>
                <div class="text-[9px] text-slate-500 mt-2">${q.Jam}</div>
                <button onclick="openAIModal(${JSON.stringify(q).replace(/"/g, '&quot;')})" class="w-full mt-2 py-1 bg-neural/20 hover:bg-neural/40 rounded-lg text-[9px] font-bold text-neural border border-neural/30 transition-all uppercase tracking-widest">Analisis AI</button>
            </div>
        `;

        // Always render circle marker
        const markerOptions = {
            radius: mag * 3,
            fillColor: color,
            color: source === 'USGS' ? '#fff' : '#fff',
            weight: source === 'USGS' ? 2 : 1,
            fillOpacity: 0.6,
            dashArray: source === 'USGS' ? '5, 5' : 'none',
            className: 'interactive-quake-marker'
        };

        L.circleMarker([lat, lon], markerOptions).addTo(quakeLayer)
            .bindTooltip(`${q.Wilayah} (${q.Magnitude} M)`, { direction: 'top', offset: [0, -5], className: 'custom-tooltip' })
            .bindPopup(popupContent, { className: 'custom-popup' })
            .on('click', function () { this.openPopup(); });

        // Add Pulse Halo for Latest Quake (BMKG only for focus)
        const isLatest = source === 'BMKG' && latest && q.Jam.trim() === latest.Jam.trim() && q.Coordinates.replace(/\s/g, '') === latest.Coordinates.replace(/\s/g, '');
        if (isLatest) {
            const pulseSize = mag * 6;
            const pulseIcon = L.divIcon({
                className: 'pulse-marker',
                iconSize: [pulseSize, pulseSize],
                iconAnchor: [pulseSize / 2, pulseSize / 2]
            });
            L.marker([lat, lon], { icon: pulseIcon, zIndexOffset: -100 })
                .addTo(quakeLayer)
                .on('click', () => openAIModal(q));
        }
    });
}

function renderFeed(recent) {
    const feed = document.getElementById('alert-feed');
    if (!feed) return;
    feed.innerHTML = "";

    // Sort all quakes by time (Jam) - this is tricky with mixed formats, but let's take first 20
    recent.slice(0, 20).forEach(q => {
        const mag = parseFloat(q.Magnitude);
        const source = q.source || 'BMKG';
        const colorClass = mag >= 6 ? 'text-danger' : (mag >= 5 ? 'text-warning' : (source === 'USGS' ? 'text-neural' : 'text-accent'));
        const div = document.createElement('div');
        div.className = "p-4 bg-white/5 border border-white/5 rounded-2xl hover:bg-white/10 transition-all cursor-pointer flex items-center gap-4 relative overflow-hidden";

        div.onclick = () => {
            openAIModal(q);
            const c = q.Coordinates.split(',');
            map.flyTo([parseFloat(c[0]), parseFloat(c[1])], 7);
        };

        div.innerHTML = `
            <div class="absolute top-0 right-0 px-2 py-0.5 bg-white/5 border-l border-b border-white/10 rounded-bl-lg">
                <span class="text-[7px] font-black uppercase tracking-tighter text-slate-500">${source}</span>
            </div>
            <div class="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center border border-white/10 shrink-0">
                <span class="text-xs font-bold ${colorClass}">${q.Magnitude}</span>
            </div>
            <div class="min-w-0 flex-1">
                <p class="text-[10px] font-bold text-white truncate uppercase tracking-tighter">${q.Wilayah}</p>
                <p class="text-[8px] text-slate-500">${q.Jam} • ${q.Kedalaman}</p>
            </div>
        `;
        feed.appendChild(div);
    });
}

window.setFilter = function (f) {
    document.querySelectorAll('[id^="btn-"]').forEach(b => b.classList.remove('filter-btn-active', 'text-slate-500'));
    if (document.getElementById(`btn-${f}`)) {
        document.getElementById(`btn-${f}`).classList.add('filter-btn-active');
    }

    if (f === 'all') {
        if (!map.hasLayer(quakeLayer)) map.addLayer(quakeLayer);
        if (!map.hasLayer(weatherLayer)) map.addLayer(weatherLayer);
    } else if (f === 'quake') {
        if (!map.hasLayer(quakeLayer)) map.addLayer(quakeLayer);
        if (map.hasLayer(weatherLayer)) map.removeLayer(weatherLayer);
    } else if (f === 'weather') {
        if (map.hasLayer(quakeLayer)) map.removeLayer(quakeLayer);
        if (!map.hasLayer(weatherLayer)) map.addLayer(weatherLayer);
    }
};

window.focusIndonesia = function () {
    map.flyTo([-2.5, 118], 5, {
        duration: 1.5,
        easeLinearity: 0.25
    });
};

window.toggleTheme = function () {
    const isDark = document.documentElement.classList.toggle('dark');
    const icon = document.getElementById('theme-icon');
    const textLabel = document.getElementById('theme-text');

    if (icon) {
        icon.setAttribute('data-lucide', isDark ? 'moon' : 'sun');
        icon.className = `w-4 h-4 ${isDark ? 'text-neural' : 'text-amber-500'}`;
    }
    if (textLabel) textLabel.innerText = isDark ? 'Dark Mode' : 'Light Mode';

    lucide.createIcons();
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
};

document.getElementById('chat-input')?.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleChat();
});

// --- LOCAL PREDICTION ENGINE ---
async function updateLocalNowcast() {
    const statusEl = document.getElementById('nowcast-status');
    const scoreEl = document.getElementById('nowcast-score');
    const recEl = document.getElementById('nowcast-rec');

    if (!userCoords || allQuakesData.length === 0) return;

    // Find nearest earthquake
    const sorted = [...allQuakesData].map(q => {
        const [qLat, qLon] = q.Coordinates.split(',').map(parseFloat);
        return { ...q, dist: calculateDistance(userCoords.lat, userCoords.lon, qLat, qLon) };
    }).sort((a, b) => a.dist - b.dist);

    const nearest = sorted[0];
    const mag = parseFloat(nearest.Magnitude);
    const dist = nearest.dist;

    // Simplified Risk Calculation
    // Risk increases with magnitude and decreases with distance
    // Example: Mag 6 at 100km is riskier than Mag 4 at 10km for regional assessment
    let riskScore = (mag * 15) - (dist / 10);
    riskScore = Math.max(0, Math.min(100, riskScore));

    // Weather Factor
    const weatherVal = document.getElementById('stat-weather-val').innerText.toLowerCase();
    const isRaining = weatherVal.includes('hujan') || weatherVal.includes('petir') || weatherVal.includes('gerimis');

    if (isRaining && riskScore > 30) riskScore += 10; // Extra risk for landslides/slippery paths
    riskScore = Math.min(100, riskScore);

    // Update UI
    scoreEl.innerText = `${riskScore.toFixed(0)}%`;

    if (riskScore < 30) {
        statusEl.innerText = "AMAN (STABIL)";
        statusEl.className = "text-xs font-bold text-emerald-400";
        recEl.innerText = "Kondisi di wilayah Anda terpantau stabil. Tetap pantau informasi cuaca harian.";
    } else if (riskScore < 60) {
        statusEl.innerText = "WASPADA (MODERAT)";
        statusEl.className = "text-xs font-bold text-warning";
        recEl.innerText = `Aktivitas di wilayah **${nearest.Wilayah}** terdeteksi cukup dekat. Periksa tas siaga bencana Anda.`;
    } else {
        statusEl.innerText = "BAHAYA (TINGGI)";
        statusEl.className = "text-xs font-bold text-danger";
        recEl.innerText = "Risiko tinggi terdeteksi! Segera identifikasi jalur evakuasi tercepat dan tetap di area terbuka jika guncangan terasa.";
    }

    // Animate the score
    scoreEl.classList.add('animate-pulse');
    setTimeout(() => scoreEl.classList.remove('animate-pulse'), 2000);
}

// --- DYNAMIC EDUCATION CORE ---
async function fetchGeoFact() {
    const topic = GEO_TOPICS[Math.floor(Math.random() * GEO_TOPICS.length)];
    try {
        const res = await axios.get(`https://en.wikipedia.org/api/rest_v1/page/summary/${topic}`);
        return {
            title: res.data.title,
            extract: res.data.extract,
            url: res.data.content_urls.desktop.page
        };
    } catch (e) {
        return null;
    }
}

async function renderEducation() {
    const factContent = document.getElementById('geo-fact-content');
    if (!factContent) return;

    // Mixed Content Logic: 60% Wiki, 40% Local
    const useWiki = Math.random() > 0.4;
    if (useWiki) {
        const wikiData = await fetchGeoFact();
        if (wikiData) {
            factContent.innerHTML = `
                <h4 class="text-xs font-bold text-white mb-2 uppercase tracking-wider">Info Geologi: ${wikiData.title}</h4>
                <p class="text-[11px] text-slate-300 leading-relaxed italic mb-3">"${wikiData.extract}"</p>
                <a href="${wikiData.url}" target="_blank" class="text-[9px] text-accent font-bold hover:underline">Pelajari lebih lanjut (Wikipedia) →</a>
            `;
            lucide.createIcons();
            return;
        }
    }

    // Fallback to Local Knowledge
    const localFact = GEO_FACTS[Math.floor(Math.random() * GEO_FACTS.length)];
    factContent.innerHTML = `
        <h4 class="text-xs font-bold text-white mb-2 uppercase tracking-wider">Tahukah Anda?</h4>
        <p class="text-[11px] text-slate-300 leading-relaxed italic">"${localFact}"</p>
    `;
    lucide.createIcons();
}

window.onload = async () => {
    // Apply saved theme
    const savedTheme = localStorage.getItem('theme') || 'dark';
    if (savedTheme === 'light') {
        document.documentElement.classList.remove('dark');
        const icon = document.getElementById('theme-icon');
        const textLabel = document.getElementById('theme-text');
        if (icon) {
            icon.setAttribute('data-lucide', 'sun');
            icon.className = 'w-4 h-4 text-amber-500';
        }
        if (textLabel) textLabel.innerText = 'Light Mode';
    }
    await initAI();
    await fetchData();
    await updateRegionalWeather();
    await renderEducation();
    getUserLocation();

    // Intervals
    setInterval(updateLocalNowcast, 30000);
    setInterval(fetchData, 300000);
    setInterval(updateRegionalWeather, 600000);
    setInterval(renderEducation, 300000); // Rotate education every 5 mins
};
