import warnings
import mne
import os
import glob
import pandas as pd

# ✅ Suppress unit warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 📁 Path to the folder containing the EDF files
folder_path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\EDF filer"

annotation_map = {
    # Movement-related
    "Bevæger hovedet": "Movement",
    "Løfter hovedet": "Movement",
    "Rykker hovedet": "Movement",
    "Kigger mod tiltaleren": "Movement",
    "Grimasserer": "Movement",
    "Rykker hovedet og gaber": "Movement",
    "Grimasserer og løfter hø. OE spontant": "Movement",
    "Bevæger hovedet spontant": "Movement",
    "Løfter hø. OE mod tiltaler": "Movement",
    "Forsøger at løfte hø. OE": "Movement",
    "Grimasserer og kniber øjne sammen": "Movement",
    "Bevæger hovedet fra side til side": "Movement",
    "hoster og gaber - bevæger arme og ben": "Movement",
    "Bevæger hø. UE, let vippen med fingrene på hø. side": "Movement",
    "Løfter hø. OE og grimasserer": "Movement",
    "Drejer hovedet mod tiltaler": "Movement",
    "Bevæger hoevdet og rykker på skulderen": "Movement",
    "Begynder at vippe med fingrene på hø. side": "Movement",
    "Bevæger hø. OE mod tiltalere": "Movement",
    "bevæget ve. fod irregulært intermitterende": "Movement",
    "tygger spontant": "Movement",
    "Vender hovedet og kigger på tiltaleren kortvarig": "Movement",
    "Grimasserer og bevæger hovedet": "Movement",
    "bevæget hø. oe spontant": "Movement",
    "motorisk meget urolig": "Movement",
    "Bevæger hoevdet og kigger sig omkring - øjne åbne": "Movement",
    "halvåbne øjne og drejer hovedet mod venstre": "Movement",
    "Lilel ryk med hovedet": "Movement",
    "Bevæger hø. UE": "Movement",
    "Bevæger hovedet og kigger op": "Movement",
    "Lofter hovedet op mod tiltaleren": "Movement",
    "bevæger sig og rykker med hænder": "Movement",
    "Bevæger hø. OE mod ansigt": "Movement",
    "Grimasserer og løfter hø. OE": "Movement",
    "rykker hovedet": "Movement",
    "pt hikker": "Movement",
    "Hostr og bevæger sig": "Movement",
    "Lille ryk med hovedet mod tiltaleren": "Movement",

    # Artifact-related
    "Artefakt": "Artifact",
    "Muscle noise": "Artifact",
    "Kurven præget af muskelstøj": "Artifact",
    "Respirator artefakter på p3": "Artifact",
    "Loss of signal": "Artifact",
    "artefakter = pt. huster": "Artifact",
    "Kurven præget af muskelstøj og en del uro omkring pt. grundet tubegener.": "Artifact",
    "Artefact!": "Artifact",
    "en del muskelartefakter da pt. bevæger sig - usederet": "Artifact",
    "en del muskelartefakter da pt bevæger hovedet": "Artifact",
    "en del artefakter især fra respiratoren": "Artifact",
    "Elektrde rettet": "Artifact",
    "affl.": "Artifact",
    "respirator artefakter på p3": "Artifact",
    "Gumler på tuben": "Artifact",
    "EKG": "Artifact",
    "Impedance": "Artifact",
    "forsøger at rette p4": "Artifact",
    "Cz rettes": "Artifact",
    "O2 rettes": "Artifact",
    "CVK rettes": "Artifact",
    "Elektrode forsøges rettet": "Artifact",
    "Kæreste rører ved pt": "Artifact",
    "Mor rører ved pt": "Artifact",
    "Spl. taler til pt": "Artifact",
    "søster tiltaler pt": "Artifact",
    "der spilles pts egen musik": "Artifact",
    "der spilles musik på stuen": "Artifact",
    "Klinisk reaktion med øjenåbning og hoveddrejning mod tiltaleren": "Artifact",
    "Grimasserer og kniber øjne sammen spontant": "Artifact",
    "åbner kort øjne og kigger mod tiltaleren": "Artifact",
    "åbner øjne kort og kigger mod tiltaleren": "Artifact",
    "åbner øjne mod tiltaler": "Artifact",
    "Kortvarig øjenåbning": "Artifact",

    # Medication-related
    "Der gives medicin": "Medication",
    "Modazolam 5mg": "Medication",
    "Nimbex 4mg": "Medication",
    "Morphin 5mg": "Medication",
    "catapressan 150ug gives IV propofol og ultiva øges til 12ml/h": "Medication",
    "ultiva øges til 7ml/h": "Medication",
    "nimbex 4mg givet": "Medication",
    "gives clonidin": "Medication",
    "får medicin af spl": "Medication",
    "Der gives IV morphin grundet tubegener": "Medication",
    "clonidin 75ug": "Medication",
    "ultiva skrues ned til 5ml/h fra 7ml/h": "Medication",
    "der gives medicin i sonde": "Medication",
    "Der gives IV medicin (Oxynorm 5mg)": "Medication",
    "clonidin150ug": "Medication",
    "ultiva 7ml/t - propofol stoppet for flere dage siden": "Medication",
    "fentanyl 1,5ml/t, midazolam stoppet kl. 10": "Medication",
    "får medicin trandate": "Medication",
    "gives trandate": "Medication",
    "pt skal have it vanco - afventer dette": "Medication",
    "der gives iv furix": "Medication",
    "stesolid 5mg gentages": "Medication",
    "der gived 2,5mg diazepam": "Medication",
    "gives 5mg morphin": "Medication",
    "nimbex igen 4mg": "Medication",
    "Propofol og ultiva 8ml/t": "Medication",
    "16mg temesta": "Medication",
    "bolus ultiva 2ml/t": "Medication",
    "Ultiva skrues ned til 5ml/h fra 7ml/h": "Medication",
    "der gives medicin og ernæring i sonde": "Medication",
    "5mg stesolid": "Medication",
    "Ultiva 18ml/t": "Medication",
    "Ultiva 3ml/h": "Medication",
    "gives vand i sonde og morphin10mg": "Medication",

    # Seizure-related
    "Seizure": "Seizure",
    "Quasirytmiske GSWs": "Seizure",
    "episode med rystelser i begge OE og øjenåbning med øjendrejning opad": "Seizure",
    "obs anfaldsopbygning - samtidig intermitterende ryk sv.t. hø. kind": "Seizure",

}

annotation_map.update({
    # Movement-related
    "Løfter hø. OE mod tiltalere": "Movement",
    "Bevæger hø. Oe mod tiltaler": "Movement",
    "Bevæger hovedet og kigger op": "Movement",
    "Lofter hovedet op mod tiltaleren": "Movement",
    "bevæger sig og rykker med hænder": "Movement",
    "Bevæger hø. OE mod ansigt": "Movement",
    "Grimasserer og løfter hø. OE": "Movement",
    "rykker hovedet": "Movement",
    "pt hikker": "Movement",
    "Lilel ryk med hovedet": "Movement",
    "drejer hovedet mod tiltaler": "Movement",
    "bevæger hovedet og runker panden": "Movement",
    "Hostr og bevæger sig": "Movement",

    # Artifact-related
    "Artefact": "Artifact",
    "Kurven præget af muskelstøj og en del uro omkring pt. grundet tubegener.": "Artifact",
    "Elektrde rettet": "Artifact",
    "affl.": "Artifact",
    "respirator artefakter på p3": "Artifact",
    "lowrow elektrodeer kan ikke sidde ordentligt grundet sveden.": "Artifact",
    "Kurven præget af muskelstøj og en del uro omkring pt. grundet tubegener. Ikke muligt med lowrow da pt. febril og elektroder ikke kan sidde fast!": "Artifact",
    "Spindel?": "Artifact",
    "Klinisk reaktion med øjenåbning og hoveddrejning mod tiltaleren": "Artifact",
    "Grimasserer og kniber øjne sammen spontant": "Artifact",
    "åbner kort øjne og kigger mod tiltaleren": "Artifact",
    "åbner øjne kort og kigger mod tiltaleren": "Artifact",
    "åbner øjne mod tiltaler": "Artifact",
    "Kigger kort på tiltaleren": "Artifact",
    "Rykker hovedet let mod tiltaleren": "Artifact",
    "Vedner hovedet kort op mod tiltaleren og åbner øjne kortvarigt": "Artifact",
    "Rynker i panden spontant": "Artifact",
    "Blink": "Artifact",
    "Kortvarig øjenåbning": "Artifact",
    "Åbner øjne og kigger på tiltaler": "Artifact",
    "Spontant åbne øjne": "Artifact",
    "Kigger mod tiltaler": "Artifact",
    "Setup Change": "Artifact",
    "Elektrode rettes": "Artifact",
    "o2 og p4 rettes": "Artifact",
    "p7 rettes": "Artifact",
    "Forsøger at rette T8 uden held - formentlig problem med reciever": "Artifact",
    "Der forsøget retning af elektroder": "Artifact",
    "rettes på tube": "Artifact",
    "Cz rettes": "Artifact",
    "spl tjekker sonde": "Artifact",
    "spl tager blodprøve": "Artifact",
    "spl tager prøve": "Artifact",
    "Der tages blodprøver": "Artifact",
    "Der tages A-gas": "Artifact",
    "A-gas tages": "Artifact",
    "Der lægges isposer omkring pt.": "Artifact",
    "Der gives innohep sc i abdomen": "Artifact",
    "undersøges af daniel": "Artifact",
    "stetoskoperes": "Artifact",
    "palperes i maven": "Artifact",
    "neu us slut": "Artifact",
    "Der pusles og tales omkring pt mhp. medcinopsætning": "Artifact",
    "Kæreste rører ved pt": "Artifact",
    "Mor rører ved pt": "Artifact",
    "Spl. taler til pt": "Artifact",
    "datter tiltaler pt": "Artifact",
    "berøres let af søn": "Artifact",
    "Søn siger farvel og rører ved pt.": "Artifact",
    "Pårørende ifa. papdatter og hustru hilser på pt.": "Artifact",
    "Spl og partner taler til Pt": "Artifact",
    "Spl. retter på pumpe og taler omkring pt.": "Artifact",

    # Medication-related
    "Der gives medicin": "Medication",
    "gives clonidin": "Medication",
    "får medicin af spl": "Medication",
    "clonidin 75ug": "Medication",
    "ultiva skrues ned til 5ml/h fra 7ml/h": "Medication",
    "der gives medicin i sonde": "Medication",
    "5mg stesolid": "Medication",
    "der gives iv furix": "Medication",
    "ultiva øges igen til 7ml/h": "Medication",
    "der gives iv medicin": "Medication",
    "gives 2.5 mg stesolid": "Medication",
    "16mg temesta": "Medication",
    "bolus ultiva 2ml/t": "Medication",
    "ultiva 8ml/t genopstartet": "Medication",
    "4mg nimbex": "Medication",
    "1mg rivotril": "Medication",
    "stesolid 5 mg igen": "Medication",

    # Seizure-related
    "Seizure": "Seizure",
    "quasirytmiske GSWs": "Seizure",
    "episode med rystelser i begge OE og øjenåbning med øjendrejning opad": "Seizure",
    "obs anfaldsopbygning - samtidig intermitterende ryk sv.t. hø. kind": "Seizure",
    
    # Resting State
    "Ro": "Resting",
    "Rolig periode": "Resting",
    "øjne lukkede under optagelsen": "Resting",
    "Øjne lukket under hele optagelsen": "Resting",
    "usederet - lukkede øjne": "Resting",
    "Ro - lukkede øjne og rolig respiration": "Resting",
    "faldet helt til ro efter omlejring - formentlig søvn": "Resting",

    # Procedure-related
    "Optagelse slut": "Procedure",
    "Moshgans cEEG projekt herfra": "Procedure",
    "Projekt EEG herfra": "Procedure",

    # Noise-related
    "hosten": "Noise",
    "hoste": "Noise",
    "Host": "Noise",
    "gumler": "Noise",
    "stuegang": "Noise",
    "larm fra håndværkere": "Noise",
    "musik": "Noise",
    "ambulance lyd": "Noise",
    "Obs": "Noise",
    "Detections Inactive": "Noise"
})


annotation_map.update({
    # Movement-related
    'bevæger hovedet': 'Movement',
    'Løfter hø. OE': 'Movement',
    'Bevæger hø. Oe mod tiltaler': 'Movement',
    'Bevæger hoevdet fra side til side': 'Movement',
    'Løfter hø. OE mod tiltaleren': 'Movement',
    'kniber øjne sammen og løfter hø. OE let - uden stimuli': 'Movement',
    'Pt. generelt urolig, bevæger arme og ben ofte spontant': 'Movement',

    # Artifact-related
    'Kurven præget af muskelstøj og en del uro omkring pt. grundet tubegener. Ikke muligt med  lowrow da pt. febril og elektroder ikke kan sidde fast!': 'Artifact',
    'en del muskelartefektaer da pt tygger på tuben. Sederet med remifentanil 15-17ml/h': 'Artifact',
    'affl.': 'Artifact',
    'kæreste rører ved pt': 'Artifact',
    'mor rører ved pt': 'Artifact',
    'kæreste synger for pt': 'Artifact',
    'Der tales direkte til pt. af stuegangsgående læge': 'Artifact',
    'datter ankommet og hilser på far': 'Artifact',
    'Kæreste og mor hilser på pt': 'Artifact',
    'Pårørende ifa. papdatter og hustru hilser på pt. og taler omkring ham, rører ham let på låret - ikke del af egentlige stimulationer til projekt.': 'Artifact',
    'Ægtefælle og spl. forsøger at rette pt. op i stolen': 'Artifact',
    'Åbner øjne og kigger på tiltaler': 'Artifact',
    'klinisk reaktion med øjenåbning og hoveddrejning mod tiltaleren': 'Artifact',
    'Vedner hovedet kort op mod tiltaleren og åbner øjne kortvarigt': 'Artifact',
    'Blink': 'Artifact',
    'Rynker panden': 'Artifact',
    'Grimasserer og rynker pande': 'Artifact',
    'Suges': 'Artifact',
    'pt suges': 'Artifact',
    'suges i munden': 'Artifact',
    'Suges igen': 'Artifact',
    'sugning': 'Artifact',
    'electrode rettes': 'Artifact',
    'p7 rettes': 'Artifact',
    'o rettes': 'Artifact',
    'Cz rettes': 'Artifact',

    # Medication-related
    'der gives 2.5 mg stesolid': 'Medication',
    'der gives medicin': 'Medication',
    'nimbex 4mg gentages': 'Medication',
    'nimbex 4mg': 'Medication',
    'der gives IT vanco': 'Medication',
    'der gives 5 mg stesolid': 'Medication',
    'fentanyl stoppet': 'Medication',
    '2,5mg diazepam gentages': 'Medication',
    'remifentanil sænket fra 8 til 5ml/t': 'Medication',
    'Medicin gives i sonde': 'Medication',

    # Seizure-related
    'Seizure': 'Seizure',
    'quasirytmiske GSWs': 'Seizure',

    # Resting State
    'Øjne lukkede': 'Resting',
    'usederet, lukkede øjne under hele optagelsen. Rynker pande ofte og indimellem hoveddejning.': 'Resting',
    'faldet helt til ro efter omlejring - formentlig søvn': 'Resting',
    'Rolig musik på gangen': 'Resting',

    # Procedure-related
    'Projekt EEG herfra': 'Procedure',
    'Moshgans projektoptagelse slut': 'Procedure',
    'undersøgelse slut': 'Procedure',

    # Noise-related
    'stuegang': 'Noise',
    'larm fra håndværkere': 'Noise',
    'lavfrekvent': 'Noise',
    'Graphoelement': 'Noise',

    # Setup-related
    'Montage is now: Long': 'Setup Change',
    'Montage is now: Db Banan': 'Setup Change',
    'Setup Change': 'Setup Change',

})

annotation_map.update({
    # Movement-related
    "Bevæger hø. OE mod tiltaler": "Movement",
    "Bevæger hø. Oe mod tiltalere": "Movement",
    "Grimmasserer og kniber øjne sammen": "Movement",
    "Grimmasserer og kniber øjne sammen spontant": "Movement",
    "Kniber øjene sammen og løfter hø. OE let - uden stimuli": "Movement",
    "hosten og gaben - bevæger arme og ben": "Movement",
    "Omlejres": "Movement",

    # Artifact-related
    "pt. hoster og bliver suget": "Artifact",
    "spindel?": "Artifact",
    "hosteanfald": "Artifact",
    "Kurven præget af muskelstøj og en del uro omkring pt. grundet tubegener. Ikke muligt med  lowrow da pt. febril og elektroder ikke kan sidde fast!": "Artifact",
    "der tales på stuen": "Artifact",
    "søn ankommet og hilser": "Artifact",
    "kæreste synger for pt": "Artifact",
    "der spilles musik for pt": "Artifact",
    "mater rører ved pt": "Artifact",
    "beøres af spl": "Artifact",
    "Pt. usederet. en del muskelartefakter under optagelsen. Åbner promte øjne under stimulationer, i øvrigt lukkede øjne ved hvile.": "Artifact",
    "suges": "Artifact",
    "Suges": "Artifact",
    "suges igen": "Artifact",
    "skal suges": "Artifact",
    "suges - hoster og grimasserere": "Artifact",
    "sugning": "Artifact",
    "der tages blodprøver": "Artifact",
    "der tages a-gas": "Artifact",
    "Retter elektrode": "Artifact",
    "der sættes iv nacl op": "Artifact",
    "Vatpind næsebor": "Artifact",
    "Vatpind igen": "Artifact",
    "neurologisk undersøgele": "Artifact",
    "der rodes ved pt": "Artifact",
    "Der forsøges at rette på enkelte elektroder": "Artifact",
    "der rettes på elektroder": "Artifact",

    # Response-related
    "Øjne åbnes": "Response",
    "åbner kort øjnene": "Response",
    "Kniber øjene sammen og løfter hø. OE let - uden stimuli": "Response",
    "klinisk reaktion med øjenåbning og hoveddrejning mod tiltaleren": "Response",

    # Medication-related
    "Får medicin af spl": "Medication",
    "der gives medicin": "Medication",
    "clonidin": "Medication",
    "2,5mg morohin guvet": "Medication",
    "ultiva reduceret til 4ml/t": "Medication",
    "nimbex 4mg gentages": "Medication",
    "får taget blodprøver - der tales omkring pt.": "Medication",

    # Seizure-related
    "Seizure": "Seizure",

    # Resting State
    "Usederet": "Resting",
    "Resting": "Resting",
    "lukkede øjne - usederet igennem flere dage": "Resting",
    "øjne lukkede hele tiden": "Resting",
    "Lidt baggrundssnak fra spl. under denne resting periode": "Resting",

    # Noise-related
    "der er stuegang og satle med larm fra bl.a. telefon": "Noise",
    "Hosten": "Noise",
    "host": "Noise",
    "Tyggen og gaben": "Noise",
    "Gaber og Hoster": "Noise",

    # Setup-related
    "Setup Change": "Setup Change",
    "Annotation": "Setup Change"
})

annotation_map.update({
    # Movement-related
    "pt. reagerer på nålepåsætning ved grimasseren og hovedbevægelse, pt. er usederet siden igår.": "Movement",
    "grimmasserer og kniber øjne sammen spontant": "Movement",
    "+afværgen": "Movement",
    "synker": "Movement",

    # Artifact-related
    "hoster": "Artifact",
    "Hoster": "Artifact",
    "Hoster spontant": "Artifact",
    "spl taler omkring pt ifm. medicingivning": "Artifact",
    "der tales omkring pt": "Artifact",
    "Rynker på panden": "Artifact",
    "sonde": "Artifact",
    "tørres om munden af spl": "Artifact",
    "tørres om munden": "Artifact",
    "tørres på panden af hustru": "Artifact",
    "tørres i ansigt af hustru": "Artifact",
    "få taget blodprøve": "Artifact",
    "få taget blodprøver": "Artifact",
    "der sættes is op grundet feber": "Artifact",

    # Medication-related
    "Usederet, men får catapressan 075ug PN som er givet lige inden EEG": "Medication",
    "furix": "Medication",
    "Nimbex 4mg gentages": "Medication",

    # Seizure-related
    "Seizure": "Seizure",

    # Resting State
    "Resting": "Resting",
    "ingen ssedation - fået 10mg morphin inden optagelse": "Resting",

    # Noise-related
    "hø F": "Noise",

    # Setup-related
    "Setup Change": "Setup Change"
})


# ✅ Load the ID-to-Tiltale mapping from the file
mapping_file = "L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\tiltale_xy_map.xlsx"
mapping_df = pd.read_excel(mapping_file, sheet_name='Ark1')

# ✅ Create a dictionary for fast lookup: {ID: (Tiltale X, Tiltale Y)}
id_to_tiltale_map = dict(zip(mapping_df['ID'], zip(mapping_df['Tiltale X'], mapping_df['Tiltale Y'])))

# ✅ Find all .edf files in the folder
edf_files = glob.glob(os.path.join(folder_path, "*.edf"))

if not edf_files:
    print("❌ No EDF files found in the folder.")
else:
    print(f"✅ Found {len(edf_files)} EDF files in {folder_path}\n")

# ✅ Initialize summary stats
total_files_processed = 0
total_resting = 0
total_medical = 0
total_familiar = 0
total_unmapped_annotations = set()
global_unmapped_annotations = set()

# 🔄 Loop through each file
for file in edf_files:
    try:
        print(f"\n🔎 Reading file: {file}")

        # ✅ Extract the ID from the filename (first 4 characters)
        file_id = os.path.basename(file)[:4]
        
        # ✅ Lookup the corresponding Tiltale X and Y values
        tiltale_x, tiltale_y = id_to_tiltale_map.get(file_id, (None, None))
        if not tiltale_x or not tiltale_y:
            print(f"⚠️ No mapping found for file ID: {file_id}")
            continue
        
        print(f"➡️ File ID: {file_id}, Tiltale X: {tiltale_x}, Tiltale Y: {tiltale_y}")

        # ✅ Load the raw EDF file
        raw = mne.io.read_raw_edf(file, preload=True)

        # ✅ If no annotations, skip the file
        if len(raw.annotations) == 0:
            print("⚠️ No annotations found in this file.")
            continue
        
        # ✅ Rename channels (DO NOT DROP)
        channels = raw.info.ch_names
        channels_rename = [i.replace('EEG ', '') for i in channels]
        channels_rename = [i.replace('-REF', '') for i in channels_rename]
        chan_dict = dict(zip(channels, channels_rename))
        mne.rename_channels(raw.info, chan_dict)
        print(f"✅ Renamed channels in {file}")

        # ✅ Set channel types for non-EEG channels to avoid montage conflicts
        non_eeg_channels = ['ECG EKG', 'Photic', 'Pulse Rate', 'IBI', 'Bursts', 'Suppr']
        for ch in non_eeg_channels:
            if ch in raw.info['ch_names']:
                raw.set_channel_types({ch: 'misc'})

        # ✅ Set the EEG montage (ignore missing channels)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore")
        print(f"✅ Montage set for {file}")

        # ✅ Create a list to store mapped annotations
        mapped_annotations = []
        unmapped_annotations = []

        for ann in raw.annotations:
            # ✅ Convert annotation description to string (handles byte strings)
            description = ann['description']
            if isinstance(description, bytes):
                description = description.decode("utf-8", errors="ignore")

            # ✅ Try to map the annotation using the existing annotation map
            mapped_description = annotation_map.get(description.strip(), description.strip())

            # ✅ Apply the Tiltale mapping
            if mapped_description == "Tiltale-X":
                mapped_description = tiltale_x
            elif mapped_description == "Tiltale-Y":
                mapped_description = tiltale_y

            if mapped_description == description.strip():
                unmapped_annotations.append(description.strip())

            mapped_annotations.append(
                {
                    'onset': ann['onset'],
                    'duration': ann['duration'],
                    'description': mapped_description
                }
            )
        
        # ✅ Overwrite annotations with mapped ones
        corrected_annotations = mne.Annotations(
            onset=[a['onset'] for a in mapped_annotations],
            duration=[a['duration'] for a in mapped_annotations],
            description=[a['description'] for a in mapped_annotations]
        )
        raw.set_annotations(corrected_annotations)
        # raw.plot(block = True)

        # ✅ Count annotations (Resting, Medical, Familiar)
        count_resting = sum(1 for ann in corrected_annotations.description if ann == "Resting")
        count_medical = sum(1 for ann in corrected_annotations.description if ann == "Medical voice")
        count_familiar = sum(1 for ann in corrected_annotations.description if ann == "Familiar voice")

        print(f"   ➡️ Resting: {count_resting}, Medical: {count_medical}, Familiar: {count_familiar}")

        # ✅ Extract events from annotations
        events_from_annot, event_dict = mne.events_from_annotations(raw)
        print(f"   ➡️ Extracted events: {event_dict}")

        # ✅ Save New File with "_mapped" suffix
        mapped_file = file.replace(".edf", "_mapped.fif")
        raw.save(mapped_file, overwrite=True)
        print(f"✅ Saved mapped file to: {mapped_file}")

        # ✅ Report Unmapped Annotations
        if unmapped_annotations:
            print(f"⚠️ Unmapped annotations found: {set(unmapped_annotations)}")

        # ✅ Update summary stats
        total_files_processed += 1
        total_resting += count_resting
        total_medical += count_medical
        total_familiar += count_familiar
        total_unmapped_annotations.update(unmapped_annotations)

        # ✅ Store unmapped annotations for final report
        global_unmapped_annotations.update(unmapped_annotations)

    except Exception as e:
        print(f"\n❌ Failed to process {file}: {e}\n")

# ✅ Generate Summary Report
print("\n📊 Summary Report:")
print(f"➡️ Total files processed: {total_files_processed}")
print(f"➡️ Total resting annotations: {total_resting}")
print(f"➡️ Total medical annotations: {total_medical}")
print(f"➡️ Total familiar voice annotations: {total_familiar}")
if total_unmapped_annotations:
    print(f"➡️ Total unmapped annotations: {len(total_unmapped_annotations)}")
    print(f"   ➡️ Unmapped annotations: {total_unmapped_annotations}")

# ✅ Print global unmapped annotations at the end
if global_unmapped_annotations:
    print("\n⚠️ 🚨 FINAL LIST OF UNMAPPED ANNOTATIONS 🚨")
    print(f"{global_unmapped_annotations}")

print("\n🚀 All files processed!")