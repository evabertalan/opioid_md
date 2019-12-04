# UPDATED version of the water within protein script, here it recalculatesthe membrane movement
#execute: vmd -dispdev text -e water_within_protein_update_membrane.tcl -args CODE FILELIST PARAMETERS> water_count.out
# e.g: FOLDER_NAME = 6b73B
# e.g: PARAMETERS = 11 9 15 13 12 18 5

set code [lindex $argv 0]

mol new ../../$code/results/step5_assembly.xplor_ext.psf type psf
mol off top
set first_frame 0
set last_frame -1

set files [lindex $argv 1]
set filelist [split [lindex $files 0]]
set nf [llength $filelist]

set distance [lindex $argv 2]

for { set i 0 } { $i <= $nf } { incr i } {
    set file_name [lindex $filelist $i]
    puts file_name
    set crnt_file [glob $file_name-pbc.dcd]
    puts crnt_file
    
	animate read dcd $crnt_file beg 0 end -1 waitfor all

	set num_steps [molinfo top get numframes]
        
    set out_file $crnt_file-warter_count_membrane_updated.txt
    
	set fid [open $out_file w]

	for {set frame 0} {$frame < $num_steps} {incr frame} {
        set upmemb [atomselect top "(resname POPC and within 5 of protein and z>10) and oxygen" frame $frame]
        set upzcoord [$upmemb get {z}]
        set z [expr (([join $upzcoord +]) / [llength $upzcoord]) -1 ] 

        set downmemb [atomselect top "(resname POPC and within 5 of protein and z<-10) and oxygen" frame $frame]
        set downzcoord [$downmemb get {z}]
        set _z [expr ([join $downzcoord +]) / [llength $downzcoord]] 

        set selection "(water within $distance of protein) and (z<$z and -z<$_z) and oxygen"

        # puts "Frame: $frame"
        set a [atomselect top $selection frame $frame]
        set num [$a num]
        puts $fid "$frame $num"
        $a delete
        $selection delete
	}
	#Close written files
	close $fid

	#set first_frame [expr {$first_frame + $num_steps}]
    animate delete all
}

puts "Done"
quit
