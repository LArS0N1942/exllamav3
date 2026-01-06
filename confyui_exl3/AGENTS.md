# AGENTS.md — confyui_exl3

Tento soubor platí pro celý strom `confyui_exl3/`.

## Účel modulu
- Balíček `confyui_exl3` obsahuje ComfyUI uzly pro práci s modely `exllamav3` včetně multimodálních (vision) embeddingů.
- Primární soubor je `confyui_exl3/__init__.py`, který definuje uzly, preset knihovny a pomocné funkce.

## Struktura a orientace v kódu
- `__init__.py`
  - Definice tříd uzlů (např. `Exl3ModelLoader`, `Exl3ImageCaptioner`, `Exl3PromptWriter`, `Exl3ClipPromptBuilder`).
  - Presety promptů: `TEXT_PRESETS`, `IMAGE_CAPTION_PRESETS`, `CLIP_PROMPT_PRESETS`.
  - Pomocné funkce: `_build_sampler`, `_tensor_to_pil`, `_resolve_preset`, `_generate_text`.

## Styl a konvence
- Dodržuj stávající styl souboru (PEP 8, srozumitelné názvy, minimální duplikace).
- Preferuj explicitní pojmenování parametrů při volání funkcí s mnoha argumenty.
- Nezaváděj try/catch bloky okolo importů.
- Nepřidávej nové závislosti bez jasné potřeby a popisu.

## Práce s embeddingy
- `Generator.generate` očekává pro jeden prompt seznam `list[MMEmbedding]`.
- Neposílej vnořený seznam (`list[list[MMEmbedding]]`), pokud `Generator.generate` interně provádí balení pro single prompt.
- Pokud vrstva vision vrátí jeden embedding, obal ho na `list` až na místě získání (`embedding_list = [embedding]`).

## Presety a prompty
- Presety definuj jako slovníky se srozumitelnými klíči; používej jazykově konzistentní popisy.
- V případě nových presetů vždy aktualizuj jejich použití v `_resolve_preset`.

## Kódové změny
- Každou změnu udrž co nejmenší a cílenou.
- Pokud měníš chování uzlu, uveď důvod v komentáři nebo v popisu změny.

## Testování
- Pokud je to možné, ověř funkčnost alespoň ručním průchodem kritických cest:
  - Načtení modelu s vision komponentou.
  - Captioning s obrázkem a ověření, že embeddingy nejsou vnořené.
- V případě nemožnosti testů to explicitně uveď ve výstupu.

## Dokumentace
- Změny v chování uzlů stručně popiš v PR summary.
- Neupravuj globální README, pokud to není výslovně vyžádáno.
