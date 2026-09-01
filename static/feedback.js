/**
 * feedback.js
 * Widget de confiance (quanti) + commentaire (quali) lié à chaque diagnostic.
 *
 * Astuce : plutôt que de modifier script.js pour lui faire exposer
 * historique_id, on intercepte globalement fetch() et on repère
 * nous-mêmes les appels vers /predict. Comme ça, script.js n'a
 * besoin d'aucune modification.
 */

(function () {
  let currentHistoriqueId = null;
  let selectedTrustScore = 0;

  // --- Interception de fetch('/predict') ---------------------------------
  const fetchOriginal = window.fetch;
  window.fetch = async function (...args) {
    const response = await fetchOriginal.apply(this, args);

    const url = typeof args[0] === "string" ? args[0] : (args[0] && args[0].url) || "";
    if (url.includes("/predict")) {
      // On clone la réponse pour ne pas consommer le body dont script.js a besoin
      response
        .clone()
        .json()
        .then((data) => {
          if (data && data.historique_id) {
            currentHistoriqueId = data.historique_id;
            resetFeedbackWidget();
          }
        })
        .catch(() => {
          /* reponse non-JSON ou erreur : on ignore silencieusement */
        });
    }

    return response;
  };

  // --- UI ------------------------------------------------------------------
  function resetFeedbackWidget() {
    selectedTrustScore = 0;
    const stars = document.querySelectorAll("#trustStars span");
    stars.forEach((s) => (s.textContent = "☆"));
    const comment = document.getElementById("feedbackComment");
    if (comment) comment.value = "";
    const merci = document.getElementById("feedbackMerci");
    if (merci) merci.hidden = true;
    const submit = document.getElementById("feedbackSubmit");
    if (submit) submit.disabled = false;
    if (comment) comment.disabled = false;
  }

  function initFeedbackWidget() {
    const stars = document.querySelectorAll("#trustStars span");
    stars.forEach((star) => {
      star.addEventListener("click", () => {
        selectedTrustScore = parseInt(star.dataset.score, 10);
        stars.forEach((s) => {
          s.textContent = parseInt(s.dataset.score, 10) <= selectedTrustScore ? "★" : "☆";
        });
      });
    });

    const submitBtn = document.getElementById("feedbackSubmit");
    if (!submitBtn) return;

    submitBtn.addEventListener("click", async () => {
      if (!selectedTrustScore) {
        alert("Choisis d'abord un niveau de confiance (1 à 5 étoiles).");
        return;
      }
      if (!currentHistoriqueId) {
        alert("Aucun diagnostic récent trouvé. Établis d'abord un diagnostic.");
        return;
      }

      const commentaire = document.getElementById("feedbackComment").value;

      try {
        const res = await fetch("/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            historique_id: currentHistoriqueId,
            trust_score: selectedTrustScore,
            commentaire: commentaire,
          }),
        });

        if (res.ok) {
          document.getElementById("feedbackMerci").hidden = false;
          submitBtn.disabled = true;
          document.getElementById("feedbackComment").disabled = true;
        } else {
          const err = await res.json().catch(() => ({}));
          alert("Erreur lors de l'envoi du feedback : " + (err.error || res.status));
        }
      } catch (e) {
        alert("Erreur réseau lors de l'envoi du feedback.");
      }
    });
  }

  document.addEventListener("DOMContentLoaded", initFeedbackWidget);
})();