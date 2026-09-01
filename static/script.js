const chips = Array.from(document.querySelectorAll(".chip"));
const search = document.getElementById("search");
const countEl = document.getElementById("count");
const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");
const resultsSection = document.getElementById("results");
const resultsRow = document.getElementById("resultsRow");
const resultsCount = document.getElementById("resultsCount");
const errorBox = document.getElementById("errorBox");

let selected = new Set();

function updateCount(){
  countEl.textContent = selected.size;
  predictBtn.disabled = selected.size === 0;
}
updateCount();

chips.forEach(chip => {
  chip.addEventListener("click", () => {
    const val = chip.dataset.value;
    if (selected.has(val)){
      selected.delete(val);
      chip.classList.remove("is-selected");
    } else {
      selected.add(val);
      chip.classList.add("is-selected");
    }
    updateCount();
  });
});

search.addEventListener("input", () => {
  const q = search.value.trim().toLowerCase();
  chips.forEach(chip => {
    const label = chip.dataset.label.toLowerCase();
    chip.classList.toggle("is-hidden", q.length > 0 && !label.includes(q));
  });
});

clearBtn.addEventListener("click", () => {
  selected.clear();
  chips.forEach(chip => chip.classList.remove("is-selected"));
  updateCount();
  resultsSection.hidden = true;
  errorBox.hidden = true;
});

predictBtn.addEventListener("click", async () => {
  errorBox.hidden = true;
  predictBtn.disabled = true;
  predictBtn.textContent = "Analyse en cours…";

  try{
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symptomes: Array.from(selected) })
    });
    const data = await res.json();
    window.currentHistoriqueId = data.historique_id;

    if (!res.ok){
      errorBox.textContent = data.error || "Une erreur est survenue.";
      errorBox.hidden = false;
      return;
    }

    resultsRow.innerHTML = "";
    data.resultats.forEach(r => {
      const card = document.createElement("div");
      card.className = "result-card";

      const top3Html = r.top3.map((t, idx) => `
        <div class="top3-item ${idx === 0 ? 'top3-item--first' : ''}">
          <span class="top3-item__rank">${idx + 1}</span>
          <span class="top3-item__name">${t.maladie}</span>
          ${t.confiance !== null ? `<span class="top3-item__pct">${t.confiance}%</span>` : ""}
        </div>
      `).join("");

      const explicationHtml = r.explication ? `
        <div class="explication">
          <div class="explication__label">Symptômes déterminants</div>
          ${r.explication.map(e => `
            <div class="explication__item">
              <span>${e.symptome}</span>
              <span class="explication__poids">${e.poids}</span>
            </div>
          `).join("")}
        </div>
      ` : "";

      card.innerHTML = `
        <div class="result-card__model">${r.modele}</div>
        <div class="result-card__top3">${top3Html}</div>
        ${explicationHtml}
      `;
      resultsRow.appendChild(card);
    });

    resultsCount.textContent = data.nb_symptomes;

    const consensusBadge = document.getElementById("consensusBadge");
    const c = data.consensus;
    const niveau = c.nb_accord === c.nb_total ? "consensus--fort"
      : c.nb_accord >= c.nb_total / 2 ? "consensus--moyen"
      : "consensus--faible";
    consensusBadge.className = "consensus " + niveau;
    consensusBadge.textContent = `${c.nb_accord} modèle(s) sur ${c.nb_total} sont d'accord sur : ${c.maladie}`;

    resultsSection.hidden = false;
    feedbackSection.hidden = false;
    resultsSection.scrollIntoView({ behavior: "smooth", block: "nearest" });

  } catch(err){
    errorBox.textContent = "Impossible de contacter le serveur. Vérifie que Flask tourne bien.";
    errorBox.hidden = false;
  } finally {
    predictBtn.disabled = selected.size === 0;
    predictBtn.textContent = "Établir le diagnostic";
  }
});

// --- Widget de feedback ---
const feedbackSection = document.getElementById("feedbackSection");
const trustStars = Array.from(document.querySelectorAll(".trust-star"));
const feedbackComment = document.getElementById("feedbackComment");
const feedbackSubmit = document.getElementById("feedbackSubmit");
const feedbackMsg = document.getElementById("feedbackMsg");

let trustScore = 0;

trustStars.forEach(star => {
  star.addEventListener("click", () => {
    trustScore = parseInt(star.dataset.value, 10);
    trustStars.forEach(s => {
      s.classList.toggle("is-active", parseInt(s.dataset.value, 10) <= trustScore);
    });
  });
});

feedbackSubmit?.addEventListener("click", async () => {
  if (!window.currentHistoriqueId || trustScore === 0) {
    feedbackMsg.textContent = "Choisis une note avant d'envoyer.";
    feedbackMsg.hidden = false;
    return;
  }

  feedbackSubmit.disabled = true;
  feedbackSubmit.textContent = "Envoi…";

  try {
    const res = await fetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        historique_id: window.currentHistoriqueId,
        trust_score: trustScore,
        commentaire: feedbackComment.value.trim()
      })
    });
    if (!res.ok) throw new Error();

    feedbackMsg.textContent = "Merci pour ton retour !";
    feedbackMsg.hidden = false;
    feedbackSubmit.textContent = "Envoyé ✓";
  } catch {
    feedbackMsg.textContent = "Erreur lors de l'envoi.";
    feedbackMsg.hidden = false;
    feedbackSubmit.disabled = false;
    feedbackSubmit.textContent = "Envoyer le retour";
  }
});