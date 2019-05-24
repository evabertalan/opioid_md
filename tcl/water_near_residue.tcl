#execute: vmd -dispdev text -e water_near_residue.tcl -args FOLDER_NAME PARAMETERS > water_near_residue_log.out
# e.g: FOLDER_NAME = 6b73B
# e.g: PARAMETERS = 233 TRP 3


set code [lindex $argv 0]

mol new ../../$code/results/step5_assembly.xplor_ext.psf type psf
mol off top
set first_frame 0
set last_frame -1

set filelist [glob ../../$code/results/namd/step7.*_production.dcd-pbc.dcd]
set nf [llength $filelist]

set resid [lindex $argv 1]
set resname [lindex $argv 2]
set distance [lindex $argv 3]

set selection "(water within $distance of resid $resid and resname $resname) and oxygen"

for { set i 1 } { $i <= $nf } { incr i } {
	set crnt_file [glob ../../$code/results/namd/step7.${i}_production.dcd-pbc.dcd]

	animate read dcd $crnt_file beg 0 end -1 waitfor all

	set num_steps [molinfo top get numframes]
        
	#set out_file $crnt_file-warter_count.txt
    set out_file $crnt_file-warter_${distance}_of_${resid}${resname}.txt
    
	set fid [open $out_file w]

	for {set frame 0} {$frame < $num_steps} {incr frame} {
        puts "Frame: $frame"
        set a [atomselect top $selection frame $frame]
        set num [$a num]
        puts $fid "$frame $num"
        $a delete
	}
	#Close written files
	close $fid

	#set first_frame [expr {$first_frame + $num_steps}]
    animate delete all
}

puts "Done"
quit
